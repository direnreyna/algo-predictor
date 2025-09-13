# src/experiment_runner.py

from pathlib import Path
from typing import Tuple

# Компоненты
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig
from .data_preparer import DataPreparer
from .model_trainer import ModelTrainer
from .backtester import Backtester

class ExperimentRunner:
    """
    Выполняет один конкретный эксперимент от начала до конца.
    """
    def __init__(self, 
                 global_cfg: AppConfig, 
                 experiment_cfg: ExperimentConfig,
                 data_preparer: DataPreparer,
                 model_trainer: ModelTrainer,
                 backtester: Backtester):
        self.cfg = global_cfg
        self.log = AppLogger()
        self.experiment_cfg = experiment_cfg
        
        # Присваиваем переданные экземпляры
        self.data_preparer = data_preparer
        self.model_trainer = model_trainer
        self.backtester = backtester

        self.log.info(f"Класс {self.__class__.__name__} инициализирован для эксперимента.")

    def run(self) -> Tuple[dict, dict]:
        """
        Запускает полный пайплайн для одного эксперимента.
        :return: Кортеж из двух словарей: (финансовые_метрики, мл_метрики).
        """
        self.log.info(f"--- Запуск эксперимента с конфигом: {self.experiment_cfg.asset_name} | {self.experiment_cfg.feature_set_name} ---")
        
        # 1: Подготовка данных. DataPreparer сам все загрузит и обработает.
        # Этот метод должен будет вернуть пути к кеш-файлам или сами данные.
        self.data_preparer.run(experiment_cfg=self.experiment_cfg)

        # 2: Обучение модели.
        # Этот метод должен будет вернуть словарь с ML-метриками.
        ml_metrics = self.model_trainer.run(experiment_cfg=self.experiment_cfg)
        
        # 3: Финансовый бэктест.
        # Этот метод должен будет вернуть словарь с финансовыми метриками.
        financial_metrics = self.backtester.run(experiment_cfg=self.experiment_cfg)

        self.log.info("--- Эксперимент успешно завершен ---")
        return financial_metrics, ml_metrics