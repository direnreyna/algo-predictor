# src/mlflow_manager.py

import mlflow
from .app_logger import AppLogger

class MLflowRunManager:
    """
    Контекстный менеджер для централизованного управления запусками MLflow.
    Автоматически начинает и завершает run, а также обрабатывает ошибки.
    """
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.log = AppLogger()
        self.run = None

    def __enter__(self):
        """Начинает run в MLflow при входе в блок 'with'."""
        self.log.info(f"--- Начало MLflow Run: {self.run_name} ---")
        self.run = mlflow.start_run(run_name=self.run_name)
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершает run, обрабатывая статус (успех/ошибка)."""
        status = "FAILED" if exc_type is not None else "SUCCESS"
        
        if status == "FAILED":
            self.log.error(f"Run '{self.run_name}' завершился с ошибкой: {exc_val}")
        
        mlflow.set_tag("status", status)
        self.log.info(f"--- Завершение MLflow Run: {self.run_name} (Статус: {status}) ---")
        mlflow.end_run()