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

    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Инструмент управления проектом алготрейдинга.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Доступные команды")

    # Команда 'search' для поиска
    parser_search = subparsers.add_parser("search", help="Запустить поиск моделей на основе конфигурации.")
    parser_search.add_argument("--config", type=str, help="Путь к YAML-файлу с пространством поиска. Если не указан, используется путь по умолчанию из AppConfig: self.DEFAULT_SEARCH_CONFIG_PATH.")
    parser_search.add_argument("--n-trials", type=int, default=None, help="Количество итераций поиска. Переопределяет значение из YAML.")

    # Команда 'infer' для получения предсказаний
    parser_infer = subparsers.add_parser("infer", help="Сделать предсказания на новых данных с помощью обученной модели.")
    parser_infer.add_argument("--run-id", type=str, required=True, help="ID запуска в MLflow, модель из которого будет использована.")
    parser_infer.add_argument("--data-path", type=str, required=True, help="Путь к новому CSV-файлу с данными.")
    parser_infer.add_argument("--output-path", type=str, default=None, help="Опциональный путь для сохранения предсказаний в CSV.")

    # Команда 'ui' для MLflow
    subparsers.add_parser("ui", help="Запустить веб-интерфейс MLflow.")
    
    args = parser.parse_args()

    if args.command == "search":
        import mlflow
        from pathlib import Path
        from main_search import SearchOrchestrator
        from src.config_loader import ConfigLoader

        cfg = AppConfig()
        
        # Выбираем путь конфига для задания пайплайна обучения модели:
        # по-умолчанию: args.config - заданный параметром в CLI
        # если его нет, то из cfg.DEFAULT_SEARCH_CONFIG_PATH
        config_path = Path(args.config) if args.config else cfg.DEFAULT_SEARCH_CONFIG_PATH
        base_config = ConfigLoader.load_from_yaml(config_path)
        
        # Устанавливаем имя эксперимента в MLflow
        experiment_name = base_config.get("experiment_name", config_path.stem)
        mlflow.set_experiment(experiment_name)

        print(f"--- Запуск поиска моделей по конфигу: {args.config}, n_trials={args.n_trials} ---")
        orchestrator = SearchOrchestrator(base_config=base_config, experiment_name=experiment_name) 
        orchestrator.run(n_trials=args.n_trials)

    elif args.command == "infer":
        from src.inferencer import Inferencer
        from src.app_logger import AppLogger
        from pathlib import Path
        
        cfg = AppConfig()
        log = AppLogger()

        inferencer = Inferencer(cfg, log)
        predictions_df = inferencer.run(run_id=args.run_id, data_path=Path(args.data_path))
        
        if args.output_path:
            output_file = Path(args.output_path)
            predictions_df.to_csv(output_file)
            print(f"Предсказания успешно сохранены в: {output_file.resolve()}")
        else:
            print("\n--- Предсказания ---")
            print(predictions_df.to_string())
            print("-------------------\n")

    elif args.command == "ui":
        print("--- Запуск MLflow UI ---")
        cmd = ["mlflow", "ui", "--host", "0.0.0.0"]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nСервер MLflow остановлен.")

if __name__ == "__main__":
    main()