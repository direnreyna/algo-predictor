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
        self.search_space_config = base_config.get("search_space")
        self.finetune_config = base_config.get("finetune_from_run")
        # Создаем экземпляры инструментов ОДИН РАЗ
        self.data_preparer = DataPreparer(self.cfg, self.log)
        self.model_trainer = ModelTrainer(self.cfg, self.log)
        self.backtester = Backtester(self.cfg, self.log)        

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """
        Определяет пространство поиска для Optuna НА ОСНОВЕ YAML-КОНФИГА.
        Optuna будет вызывать эту функцию для генерации параметров каждого нового эксперимента.
        """
        if self.search_space_config is None:
            self.log.info("Секция 'search_space' не найдена. Запуск в режиме 'хардмод' с фиксированными параметрами.")
            return self.base_config
        
        params = {}
        # 1. Копируем все "замороженные" параметры из базового конфига
        for key, value in self.base_config.items():
            if key != "search_space":
                params[key] = value

        # Инициализируем вложенные словари
        params["model_params"] = {}
        params["train_params"] = {}

        # 2. Динамически генерируем параметры для перебора
        for param_name, config in self.search_space_config.items():
            ### # Пропускаем вложенные словари на этом уровне
            ### if param_name == "model_params": continue

            param_type = config["type"]
            value = None

            if param_type == "categorical":
                ### params[param_name] = trial.suggest_categorical(param_name, config["choices"])
                value = trial.suggest_categorical(param_name, config["choices"])
            elif param_type == "int":
                ### params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
                value = trial.suggest_int(param_name, config["low"], config["high"])
            elif param_type == "float":
                ### params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=config.get("log", False))
                value = trial.suggest_float(param_name, config["low"], config["high"], log=config.get("log", False))
            # ... можно добавить другие типы ...

            # Распределяем параметры по группам
            if param_name in ['lstm_units', 'dropout_rate', 'n_estimators', 'num_leaves']:
                params["model_params"][param_name] = value
            elif param_name in ['epochs', 'learning_rate', 'early_stopping_patience']:
                params["train_params"][param_name] = value
            else: # Остальные параметры (feature_set_name, etc.) остаются на верхнем уровне
                params[param_name] = value

        ### # Особая обработка вложенных параметров модели
        ### # Проверяем, что 'model_params' есть и является словарем
        ### model_params_space = self.search_space_config.get("model_params")
        ### if isinstance(model_params_space, dict):
        ###     # Инициализируем словарь, если его еще нет
        ###     if "model_params" not in params:
        ###         params["model_params"] = {}
        ### 
        ###     for sub_param_name, sub_config in model_params_space.items():
        ###         param_type = sub_config["type"]
        ###         # ... аналогичная логика для model_params ...
        ###         if param_type == "int":
        ###              params["model_params"][sub_param_name] = trial.suggest_int(sub_param_name, sub_config["low"], sub_config["high"])
        ###         elif param_type == "float":
        ###              params["model_params"][sub_param_name] = trial.suggest_float(sub_param_name, sub_config["low"], sub_config["high"], log=sub_config.get("log", False))
        ###         # ... и т.д.
                
        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Целевая функция для Optuna (режим ПОИСКА).
        """
        run_name = f"trial_{trial.number}"
        
        # 1. Начинаем запись в MLflow
        with MLflowRunManager(run_name=run_name) as run:
            self.log.info(f"--- Начало эксперимента (поиск): {run_name} (MLflow Run ID: {run.info.run_id}) ---")
            
            # 2. Генерируем "паспорт" эксперимента на основе предложений Optuna
            all_params = self.define_search_space(trial)

            # Получаем список полей, которые ожидает наш dataclass
            config_fields = ExperimentConfig.__annotations__.keys()
            # Фильтруем словарь, оставляя только нужные ключи
            config_params = {k: v for k, v in all_params.items() if k in config_fields}
            experiment_config = ExperimentConfig(**config_params)
            
            # Логируем параметры в MLflow
            mlflow.log_params(experiment_config.to_dict())
            trial.set_user_attr("full_params", experiment_config.to_dict())

            try:
                # 3. Создаем и запускаем "прораба", который выполнит всю грязную работу
                ### runner = ExperimentRunner(global_cfg=self.cfg, experiment_cfg=experiment_config)
                runner = ExperimentRunner(
                    global_cfg=self.cfg, 
                    experiment_cfg=experiment_config,
                    data_preparer=self.data_preparer,
                    model_trainer=self.model_trainer,
                    backtester=self.backtester
                )
                # Извлекаем параметры для передачи в runner.run()
                log_history = experiment_config.log_history_per_epoch

                financial_metrics, ml_metrics = runner.run(log_history_per_epoch=log_history) # Этот метод возвращает словари с метриками

                # 4. Логируем результаты в MLflow
                self.log.info(f"Эксперимент {run_name} завершен. Sharpe: {financial_metrics['sharpe_ratio']:.2f}")
                mlflow.log_metrics(financial_metrics)
                mlflow.log_metrics(ml_metrics)
                
                # 5. Возвращаем Optuna главную метрику для оптимизации
                # Optuna будет пытаться ее максимизировать (или минимизировать, если указать direction)
                return financial_metrics["sharpe_ratio"]

            except Exception as e:
                self.log.error(f"Эксперимент {run_name} провалился: {e}", exc_info=True)
                trial.set_user_attr("failure_reason", str(e)) # Сохраняем причину в Optuna
                raise optuna.exceptions.TrialPruned() # Говорим Optuna "убить" этот trial ##ИЗМЕНЕНО

    def _execute_single_run(self):
        """Выполняет один запуск в режиме дообучения (finetune) или хардмода."""
        
        is_finetune = self.finetune_config and self.finetune_config.get("run_id")
        run_name = "finetune_run" if is_finetune else "hard_mode_run"

        with MLflowRunManager(run_name=run_name) as run:
            self.log.info(f"--- Начало эксперимента ({run_name}): {run.info.run_id} ---")

            all_params = self.base_config
            warm_start_dict = None

            if is_finetune:
                source_run_id = self.finetune_config["run_id"]
                self.log.info(f"Загрузка параметров из исходного run_id: {source_run_id}")
                
                try:
                    client = MlflowClient()
                    source_run = client.get_run(source_run_id)
                    source_params = source_run.data.params
                    
                    # Приоритет у параметров из текущего YAML
                    merged_params = source_params.copy()

                    # Глубокое слияние словарей model_params и train_params
                    # Это позволяет переопределять отдельные ключи, а не всю секцию
                    for key in ["model_params", "train_params"]:
                        if key in all_params:
                            merged_params[key] = {**merged_params.get(key, {}), **all_params[key]}
                    
                    # Обновляем остальные верхнеуровневые параметры
                    other_params = {k: v for k, v in all_params.items() if k not in ["model_params", "train_params"]}
                    merged_params.update(other_params)
                    ### merged_params.update(all_params)

                    all_params = merged_params
                    
                    warm_start_dict = {"run_id": source_run_id}
                except Exception as e:
                    self.log.error(f"Не удалось загрузить данные из run_id '{source_run_id}': {e}")
                    raise
            
            config_fields = ExperimentConfig.__annotations__.keys()

            # Извлекаем и разделяем параметры для конструктора ExperimentConfig
            model_params = all_params.pop("model_params", {})
            train_params = all_params.pop("train_params", {})
            
            # Фильтруем оставшиеся параметры
            config_params = {k: v for k, v in all_params.items() if k in config_fields}
            
            # Собираем финальный словарь для инициализации
            final_init_params = {**config_params, "model_params": model_params, "train_params": train_params}

            ###experiment_config = ExperimentConfig(**config_params)
            experiment_config = ExperimentConfig(**final_init_params)

            mlflow.log_params(experiment_config.to_dict())

            try:
                runner = ExperimentRunner(
                    global_cfg=self.cfg,
                    experiment_cfg=experiment_config,
                    data_preparer=self.data_preparer,
                    model_trainer=self.model_trainer,
                    backtester=self.backtester,
                )
                
                financial_metrics, ml_metrics = runner.run(
                    warm_start=warm_start_dict, 
                    log_history_per_epoch=experiment_config.log_history_per_epoch
                )

                self.log.info(f"Эксперимент {run_name} завершен. Sharpe: {financial_metrics['sharpe_ratio']:.2f}")
                mlflow.log_metrics(financial_metrics)
                mlflow.log_metrics(ml_metrics)

            except Exception as e:
                self.log.error(f"Эксперимент {run_name} провалился: {e}", exc_info=True)
                raise
            
###    def _run_finetune_trial(self):
###        """Выполняет один запуск в режиме дообучения (finetune)."""
###        run_name = "finetune_trial"
###        with MLflowRunManager(run_name=run_name) as run:
###            self.log.info(f"--- Начало эксперимента: {run_name} (MLflow Run ID: {run.info.run_id}) ---")
###
###            # В режиме дообучения все параметры берутся из базового конфига
###            all_params = self.base_config
###
###            # Получаем список полей, которые ожидает наш dataclass
###            config_fields = ExperimentConfig.__annotations__.keys()
###            config_params = {k: v for k, v in all_params.items() if k in config_fields}
###            experiment_config = ExperimentConfig(**config_params)
###
###            # Логируем параметры в MLflow
###            mlflow.log_params(experiment_config.to_dict())
###
###            try:
###                runner = ExperimentRunner(
###                    global_cfg=self.cfg,
###                    experiment_cfg=experiment_config,
###                    data_preparer=self.data_preparer,
###                    model_trainer=self.model_trainer,
###                    backtester=self.backtester,
###                )
###
###                warm_start = experiment_config.finetune_from_run
###                log_history = experiment_config.log_history_per_epoch
###                
###                financial_metrics, ml_metrics = runner.run(
###                    warm_start=warm_start, log_history_per_epoch=log_history
###                )
###
###                self.log.info(f"Эксперимент {run_name} завершен. Sharpe: {financial_metrics['sharpe_ratio']:.2f}")
###                mlflow.log_metrics(financial_metrics)
###                mlflow.log_metrics(ml_metrics)
###
###            except Exception as e:
###                self.log.error(f"Эксперимент {run_name} провалился: {e}", exc_info=True)
###                # В режиме одного запуска просто выбрасываем ошибку дальше
###                raise

    def run(self, n_trials: int = 100):
        """
        Запускает весь процесс поиска или дообучения.
        """
        self.log.info("Синхронизация источников данных...")
        file_loader = FileLoader(self.cfg, self.log)
        file_loader.sync_data_sources()

        # 1. Режим Дообучения
        if self.finetune_config and self.finetune_config.get("run_id"):
            self.log.info(f"Запуск в режиме ДООБУЧЕНИЯ из run_id: {self.finetune_config['run_id']}")
            self._execute_single_run()

        # 2. Режим Хардмода (фиксированный запуск)
        elif self.base_config.get("model_params") and self.base_config.get("train_params"):
             self.log.info("Запуск в режиме ХАРДМОД (фиксированные параметры).")
             self._execute_single_run()

        # 3. Режим Поиска
        elif self.search_space_config:
            self.log.info(f"Запуск в режиме ПОИСКА. Количество экспериментов: {n_trials}")

            # Создаем "исследование" в Optuna. direction="maximize" означает, что мы хотим максимизировать результат objective
            study = optuna.create_study(direction="maximize")
            
            # Запускаем оптимизацию. Optuna будет вызывать self.objective n_trials раз
            study.optimize(self.objective, n_trials=n_trials)

            # Выводим лучшие результаты
            self.log.info("="*80)
            self.log.info("Процесс поиска завершен.")
            
            best_trial = study.best_trial
            
            self.log.info(f"Лучший результат (Sharpe): {best_trial.value:.4f}")
            self.log.info(f"Найден в эксперименте: trial_{best_trial.number}")
            
            # Ищем соответствующий MLflow Run ID
            best_run_id = None

            runs = mlflow.search_runs(experiment_names=[self.experiment_name],
                                    filter_string=f"tags.mlflow.runName = 'trial_{best_trial.number}'",
                                    max_results=1) # Добавляем для эффективности
            
            if not runs.empty:
                best_run_id = runs.iloc[0].run_id
                self.log.info(f"MLflow Run ID: {best_run_id}")
                
            self.log.info("Лучшие параметры:")

            best_params = best_trial.user_attrs.get("full_params", best_trial.params)
            for key, value in best_params.items():
                self.log.info(f"  {key}: {value}")

        # 4. Ошибка конфигурации
        else:
            raise ValueError(
                "Некорректная конфигурация: не задан run_id для дообучения, "
                "не указаны model_params/train_params для фиксированного запуска "
                "и отсутствует секция search_space для поиска."
            )

# Старая точка входа. Теперь запуск осуществляется через manage.py
### if __name__ == '__main__':
###     orchestrator = SearchOrchestrator()
###     orchestrator.run(n_trials=50) # Запускаем 50 тестовых экспериментов