# src.model_trainer.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger
from .data_saver import DataSaver        

class ModelTrainer:
    """
    Отвечает за создание, обучение и сохранение модели.

    Нюанс:
    Преобразование в OHE — это задача ModelTrainer, потому что это техническое требование модели (loss='categorical_crossentropy'), а не часть семантической подготовки данных.
    И самый лучший способ сделать это — "на лету" с помощью tf.data.Dataset.map(), как мы и сделали в нашем финальном коде DataPreparer. Моя ошибка была в том, что я поместил этот код в DataPreparer, а его место — в ModelTrainer.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.data_saver = DataSaver(cfg, log)
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
        datasets = self.data_saver.load()