# manage.py

import os
import sys
import subprocess
import argparse

from src.app_config import MLFLOW_TRACKING_URI, AppConfig
# Корректируем окружение для ВСЕГО приложения.
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

def main():
    """
    Единая точка входа для управления проектом.
    Автоматически настраивает окружение и передает управление
    нужному компоненту.
    """

    ## БЛОК ПАРСИНГА АРГУМЕНТОВ
    parser = argparse.ArgumentParser(description="Инструмент управления проектом алготрейдинга.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    # Команда 'run' для одиночного запуска
    parser_run = subparsers.add_parser("run", help="Запустить один эксперимент по файлу конфигурации.")
    parser_run.add_argument("--config", type=str, required=True, help="Путь к YAML-файлу конфигурации эксперимента.")
    
    # Команда 'search' для поиска
    parser_search = subparsers.add_parser("search", help="Запустить поиск моделей на основе конфигурации.")
    parser_search.add_argument("--config", type=str, required=True, help="Путь к YAML-файлу с пространством поиска.")
    parser_search.add_argument("--n-trials", type=int, default=100, help="Количество итераций поиска.")

    # Команда 'ui' для MLflow
    subparsers.add_parser("ui", help="Запустить веб-интерфейс MLflow.")
    
    args = parser.parse_args()

    if args.command == "run":
        # Логика для одиночного запуска (будет реализована позже)
        print(f"--- Запуск одиночного эксперимента с конфигом: {args.config} ---")
        # runner = ExperimentRunner(...)
        # runner.run()
        pass

    elif args.command == "search":
        import mlflow
        from pathlib import Path
        from main_search import SearchOrchestrator
        from src.config_loader import ConfigLoader

        config_path = Path(args.config)
        base_config = ConfigLoader.load_from_yaml(config_path)
        
        # Устанавливаем имя эксперимента в MLflow
        experiment_name = base_config.get("experiment_name", config_path.stem)
        mlflow.set_experiment(experiment_name)

        print(f"--- Запуск поиска моделей по конфигу: {args.config} ---")
        orchestrator = SearchOrchestrator(base_config=base_config, experiment_name=experiment_name) 
        #orchestrator = SearchOrchestrator(base_config=base_config)
        orchestrator.run(n_trials=args.n_trials)

    elif args.command == "ui":
        print("--- Запуск MLflow UI ---")
        cmd = ["mlflow", "ui", "--host", "0.0.0.0"]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nСервер MLflow остановлен.")

if __name__ == "__main__":
    main()