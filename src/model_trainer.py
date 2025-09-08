# src.model_trainer.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger

class ModelTrainer:
    """
    Отвечает за создание, обучение и сохранение модели.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self):
        """
        Запускает полный цикл:
        - Нарезка данных на последовательности.
        - Создание tf.data.Dataset.
        - Обучение модели.
        - Сохранение лучшей модели.
        """
        self.log.info("Запуск процесса обучения модели...")
        # ...
        pass