# main_search.py (точка входа для поиска сборок моделей)

import mlflow
import optuna
from pathlib import Path
from mlflow import MlflowClient

# Компоненты
from src.app_config import AppConfig
from src.app_logger import AppLogger
from src.mlflow_manager import MLflowRunManager
from src.experiment_runner import ExperimentRunner
from src.entities import ExperimentConfig
from src.file_loader import FileLoader
from src.data_preparer import DataPreparer
from src.model_trainer import ModelTrainer
from src.backtester import Backtester

class SearchOrchestrator:
    """
    Класс-оркестратор верхнего уровня.
    Отвечает за:
    1. Определение пространства поиска (гиперпараметры, признаки и т.д.).
    2. Запуск процесса оптимизации с помощью Optuna.
    3. Логирование каждого запуска в MLflow.
    """
    def __init__(self, base_config: dict, experiment_name: str):
        self.cfg = AppConfig()
        self.log = AppLogger()
        self.base_config = base_config
        self.experiment_name = experiment_name

        # Определяем режим работы
        self.mode = self.base_config.get("mode")
        if not self.mode:
            raise ValueError("В конфигурационном файле отсутствует обязательный ключ 'mode'.")
            
        # Извлекаем основные блоки конфигурации
        self.common_params = self.base_config.get("common_params", {})
        self.search_mode_config = self.base_config.get("search_mode", {})
        self.train_mode_config = self.base_config.get("train_mode", {})
        self.finetune_mode_config = self.base_config.get("finetune_mode", {})

        # Создаем экземпляры инструментов ОДИН РАЗ
        self.data_preparer = DataPreparer(self.cfg, self.log)
        self.model_trainer = ModelTrainer(self.cfg, self.log)
        self.backtester = Backtester(self.cfg, self.log)        

    def _generate_params_from_trial(self, trial: optuna.Trial) -> tuple[dict, dict]:
        """Генерирует model_params и train_params на основе trial."""
        model_params = {}
        train_params = {}
        search_space = self.search_mode_config.get("space", {})

        # Динамически генерируем параметры для перебора
        for group_name, params_group in search_space.items():
            if not isinstance(params_group, dict): continue
            
            for param_name, config in params_group.items():
                if not isinstance(config, dict): continue
                param_type = config.get("type")
                value = None

                if param_type == "categorical":
                    value = trial.suggest_categorical(param_name, config["choices"])
                elif param_type == "int":
                    value = trial.suggest_int(param_name, config["low"], config["high"])
                elif param_type == "float":
                    value = trial.suggest_float(param_name, config["low"], config["high"], log=config.get("log", False))
                
                if group_name == "model_params":
                    model_params[param_name] = value
                elif group_name == "train_params":
                    train_params[param_name] = value

        return model_params, train_params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Целевая функция для Optuna (режим ПОИСКА).
        """
        run_name = f"trial_{trial.number}"
        
        # 1. Начинаем запись в MLflow
        with MLflowRunManager(run_name=run_name) as run:
            self.log.info(f"--- Начало эксперимента (поиск): {run_name} (MLflow Run ID: {run.info.run_id}) ---")
            
            try:
                # 1. Загружаем "шаблон" с фиксированными параметрами из конфига
                model_params = self.search_mode_config.get("model_params", {}).copy()
                train_params = self.search_mode_config.get("train_params", {}).copy()

                # 2. Генерируем гиперпараметры для этого trial и ОБНОВЛЯЕМ ими шаблон
                trial_model_params, trial_train_params = self._generate_params_from_trial(trial)
                model_params.update(trial_model_params)
                train_params.update(trial_train_params)

                # 3. Блок умного переопределения параметров конфига для Оптуны
                # Создаем копию общих параметров, чтобы не изменять оригинал
                current_common_params = self.common_params.copy()

                # Проверяем, перебираются ли общие параметры (такие как labeling_horizon)
                # и обновляем их значения для текущего trial
                search_space = self.search_mode_config.get("space", {})
                for param_name, config in search_space.items():
                    # Ищем параметры, которые не являются словарями (т.е. лежат на верхнем уровне space)
                    if not isinstance(config, dict): continue
                    if param_name in ["model_params", "train_params"]: continue
                    
                    # Если параметр есть в common_params, Optuna его уже сгенерировала
                    if param_name in trial.params:
                        current_common_params[param_name] = trial.params[param_name]

                # 4. Собираем полный "паспорт" эксперимента
                experiment_config = ExperimentConfig(
                    common_params=current_common_params,
                    base_config=self.base_config,
                    model_params=model_params,
                    train_params=train_params,
                    log_history_per_epoch=self.search_mode_config.get("log_history_per_epoch", False)
                )

                # Логируем параметры в MLflow
                mlflow.log_params(experiment_config.to_dict())
                trial.set_user_attr("full_params", experiment_config.to_dict())

                # 5. Создаем и запускаем "прораба", который выполнит всю грязную работу
                runner = ExperimentRunner(
                    global_cfg=self.cfg, 
                    experiment_cfg=experiment_config,
                    data_preparer=self.data_preparer,
                    model_trainer=self.model_trainer,
                    backtester=self.backtester
                )
                assert self.mode is not None, "Режим 'mode' не должен быть None на этом этапе"
                financial_metrics, ml_metrics = runner.run(mode=self.mode)

                # 6. Логируем результаты и возвращаем целевую метрику
                objective_metric_name = self.search_mode_config.get("objective_metric", "sharpe_ratio")
                direction = self.search_mode_config.get("objective_direction", "maximize")
                default_value = float('inf') if direction == "minimize" else 0.0

                # Ищем метрику сначала в ML, потом в финансовых результатах
                metric_value = ml_metrics.get(objective_metric_name)
                if metric_value is None:
                    metric_value = financial_metrics.get(objective_metric_name, default_value)

                self.log.info(f"Эксперимент {run_name} завершен. {objective_metric_name.upper()}: {metric_value:.4f}")
                mlflow.log_metrics(financial_metrics)
                mlflow.log_metrics(ml_metrics)
                
                return metric_value

            except Exception as e:
                self.log.error(f"Эксперимент {run_name} провалился: {e}", exc_info=True)
                trial.set_user_attr("failure_reason", str(e)) # Сохраняем причину в Optuna
                raise optuna.exceptions.TrialPruned() # Говорим Optuna "убить" этот trial

    def _run_search(self):
        """Запускает процесс поиска с Optuna."""
        n_trials = self.search_mode_config.get("n_trials", 1)
        self.log.info(f"Запуск в режиме ПОИСКА. Количество экспериментов: {n_trials}")

        # Блок настройки хранилища SQLITE. Создаем имя для файла БД на основе имени эксперимента
        study_db_filename = f"{self.experiment_name.replace(' ', '_').lower()}.db"
        study_db_path = Path(self.cfg.LOGS_DIR) / study_db_filename
        storage_url = f"sqlite:///{study_db_path.as_posix()}"
        self.log.info(f"Используется хранилище Optuna: {storage_url}")

        direction = self.search_mode_config.get("objective_direction", "maximize")
        self.log.info(f"Направление оптимизации: {direction}")

        study = optuna.create_study(
            study_name=self.experiment_name,
            storage=storage_url,
            load_if_exists=True,
            direction=direction
        )
        study.optimize(self.objective, n_trials=n_trials)

        self.log.separator()
        self.log.info("Процесс поиска завершен.")
        best_trial = study.best_trial
        objective_metric_name = self.search_mode_config.get("objective_metric", "sharpe_ratio")
        self.log.info(f"Лучший результат ({objective_metric_name.upper()}): {best_trial.value:.4f}")
        self.log.info(f"Найден в эксперименте: trial_{best_trial.number}")
        
        best_run_id = None
        runs = mlflow.search_runs(experiment_names=[self.experiment_name],
                                filter_string=f"tags.mlflow.runName = 'trial_{best_trial.number}'",
                                max_results=1)

        # Проверяем, не пустой ли результат, независимо от его типа (list или DataFrame)
        not_empty = False
        if isinstance(runs, list):
            not_empty = len(runs) > 0
        else: # Предполагаем, что это DataFrame
            not_empty = not runs.empty

        if not_empty:
            # Получаем run_id в зависимости от типа
            if isinstance(runs, list):
                best_run_id = runs[0].info.run_id
            else: # DataFrame
                best_run_id = runs.iloc[0].run_id
            self.log.info(f"MLflow Run ID: {best_run_id}")

        self.log.info("Лучшие параметры:")

        best_params = best_trial.user_attrs.get("full_params", {})
        for key, value in best_params.items():
            self.log.info(f"  {key}: {value}")

    def _run_single_mode(self, mode_name: str, config: dict):
        """Выполняет один запуск (train или finetune)."""
        run_name = f"{mode_name}_run"
        with MLflowRunManager(run_name=run_name) as run:
            self.log.info(f"--- Начало эксперимента ({mode_name}): {run_name} (MLflow Run ID: {run.info.run_id}) ---")
            try:
                model_params = config.get("model_params", {})
                train_params = config.get("train_params", {})
                warm_start = None

                if mode_name == "finetune":
                    source_run_id = config.get("source_run_id")
                    if not source_run_id:
                        raise ValueError("Для режима 'finetune' в конфиге должен быть указан 'source_run_id'.")
                    warm_start = {"run_id": source_run_id}

                experiment_config = ExperimentConfig(
                    common_params=self.common_params,
                    base_config=self.base_config,
                    model_params=model_params,
                    train_params=train_params,
                    log_history_per_epoch=config.get("log_history_per_epoch", True)
                )

                mlflow.log_params(experiment_config.to_dict())

                runner = ExperimentRunner(
                    global_cfg=self.cfg, experiment_cfg=experiment_config,
                    data_preparer=self.data_preparer, model_trainer=self.model_trainer,
                    backtester=self.backtester
                )
                assert self.mode is not None, "Режим 'mode' не должен быть None на этом этапе"
                financial_metrics, ml_metrics = runner.run(warm_start=warm_start, mode=self.mode)

                self.log.info(f"Эксперимент {run_name} завершен. Sharpe: {financial_metrics.get('sharpe_ratio', 0.0):.2f}")
                mlflow.log_metrics(financial_metrics)
                mlflow.log_metrics(ml_metrics)
            except Exception as e:
                self.log.error(f"Эксперимент {run_name} провалился: {e}", exc_info=True)
                raise

    def run(self, n_trials: int | None = None):
        """Запускает весь процесс на основе 'mode' из конфига."""
        self.log.info("Синхронизация источников данных...")
        file_loader = FileLoader(self.cfg, self.log)
        file_loader.sync_data_sources()

        if self.mode == "search":
            # Если n_trials передан из CLI (не None), он имеет приоритет
            if n_trials is not None:
                self.search_mode_config['n_trials'] = n_trials
            self._run_search()
        elif self.mode == "train":
            self._run_single_mode("train", self.train_mode_config)
        elif self.mode == "finetune":
            self._run_single_mode("finetune", self.finetune_mode_config)
        else:
            raise ValueError(f"Неизвестный режим '{self.mode}' в конфигурационном файле.")

# Старая точка входа. Теперь запуск осуществляется через manage.py
### if __name__ == '__main__':
###     orchestrator = SearchOrchestrator()
###     orchestrator.run(n_trials=50) # Запускаем 50 тестовых экспериментов