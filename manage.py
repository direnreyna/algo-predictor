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

    # Команда 'search' для поиска
    parser_search = subparsers.add_parser("search", help="Запустить поиск моделей на основе конфигурации.")
    parser_search.add_argument("--config", type=str, required=True, help="Путь к YAML-файлу с пространством поиска.")
    parser_search.add_argument("--n-trials", type=int, default=1, help="Количество итераций поиска.")

    # Команда 'ui' для MLflow
    subparsers.add_parser("ui", help="Запустить веб-интерфейс MLflow.")
    
    args = parser.parse_args()

    if args.command == "search":
        import mlflow
        from pathlib import Path
        from main_search import SearchOrchestrator
        from src.config_loader import ConfigLoader

        config_path = Path(args.config)
        base_config = ConfigLoader.load_from_yaml(config_path)
        
        # Устанавливаем имя эксперимента в MLflow
        experiment_name = base_config.get("experiment_name", config_path.stem)
        mlflow.set_experiment(experiment_name)

        print(f"--- Запуск поиска моделей по конфигу: {args.config}, n_trials={args.n_trials} ---")
        orchestrator = SearchOrchestrator(base_config=base_config, experiment_name=experiment_name) 
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