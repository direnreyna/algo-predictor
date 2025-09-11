# main_search.py (точка входа для поиска сборок моделей)

import mlflow
import optuna
from pathlib import Path

# Компоненты
from src.app_config import AppConfig
from src.app_logger import AppLogger
from src.mlflow_manager import MLflowRunManager
# from src.experiment_runner import ExperimentRunner # Этот класс мы сейчас проектируем мысленно
from src.entities import ExperimentConfig

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

        if not self.search_space_config:
            raise ValueError("Секция 'search_space' не найдена в файле конфигурации для поиска.")

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """
        Определяет пространство поиска для Optuna НА ОСНОВЕ YAML-КОНФИГА.
        Optuna будет вызывать эту функцию для генерации параметров каждого нового эксперимента.
        """
        if self.search_space_config is None:
            return self.base_config
        
        params = {}
        # 1. Копируем все "замороженные" параметры из базового конфига
        for key, value in self.base_config.items():
            if key != "search_space":
                params[key] = value

        # 2. Динамически генерируем параметры для перебора
        for param_name, config in self.search_space_config.items():
            # Пропускаем вложенные словари на этом уровне
            if param_name == "model_params": continue

            param_type = config["type"]
            
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=config.get("log", False))
            # ... можно добавить другие типы ...
        
        # Особая обработка вложенных параметров модели
        # Проверяем, что 'model_params' есть и является словарем
        model_params_space = self.search_space_config.get("model_params")
        if isinstance(model_params_space, dict):
            # Инициализируем словарь, если его еще нет
            if "model_params" not in params:
                params["model_params"] = {}

            for sub_param_name, sub_config in model_params_space.items():
                param_type = sub_config["type"]
                # ... аналогичная логика для model_params ...
                if param_type == "int":
                     params["model_params"][sub_param_name] = trial.suggest_int(sub_param_name, sub_config["low"], sub_config["high"])
                elif param_type == "float":
                     params["model_params"][sub_param_name] = trial.suggest_float(sub_param_name, sub_config["low"], sub_config["high"], log=sub_config.get("log", False))
                # ... и т.д.
                
        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Целевая функция для Optuna. Это сердце оркестратора.
        Она выполняется для КАЖДОГО эксперимента (trial).
        """
        run_name = f"trial_{trial.number}"
        
        # 1. Начинаем запись в MLflow
        with MLflowRunManager(run_name=run_name) as run:
            self.log.info(f"--- Начало эксперимента: {run_name} (MLflow Run ID: {run.info.run_id}) ---")
            
            # 2. Генерируем "паспорт" эксперимента на основе предложений Optuna
            all_params = self.define_search_space(trial) ##ИЗМЕНЕНО

            # Получаем список полей, которые ожидает наш dataclass
            config_fields = ExperimentConfig.__annotations__.keys()
            # Фильтруем словарь, оставляя только нужные ключи
            config_params = {k: v for k, v in all_params.items() if k in config_fields}
            experiment_config = ExperimentConfig(**config_params)
            
            # Логируем параметры в MLflow
            mlflow.log_params(experiment_config.to_dict())

            try:
                # 3. Создаем и запускаем "прораба", который выполнит всю грязную работу
                # runner = ExperimentRunner(global_cfg=self.cfg, experiment_cfg=experiment_config)
                # financial_metrics, ml_metrics = runner.run() # Этот метод должен вернуть словари с метриками
                
                # --- ЗАГЛУШКА: Эмулируем работу runner ---
                import random
                if random.random() < 0.1: raise ValueError("Случайная ошибка симуляции") ##ДОБАВЛЕНО для теста
                financial_metrics = {"sharpe_ratio": random.uniform(0.5, 2.5), "max_drawdown": -random.uniform(0.05, 0.2)}
                ml_metrics = {"accuracy": random.uniform(0.5, 0.6)}
                # --- КОНЕЦ ЗАГЛУШКИ ---

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
            
    def run(self, n_trials: int = 100):
        """
        Запускает весь процесс поиска.
        """
        self.log.info(f"Запуск процесса поиска. Количество экспериментов: {n_trials}")
        
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
        for key, value in best_trial.params.items():
            self.log.info(f"  {key}: {value}")

if __name__ == '__main__':
    orchestrator = SearchOrchestrator()
    orchestrator.run(n_trials=50) # Запускаем 50 тестовых экспериментов