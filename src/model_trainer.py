# src.model_trainer.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger
from .data_saver import DataSaver        
from .entities import ExperimentConfig
from .cache_utils import get_cache_filename

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

    def run(self, experiment_cfg: ExperimentConfig) -> dict:
        """
        Запускает полный цикл:
        - Нарезка данных на последовательности.
        - Создание tf.data.Dataset.
        - Обучение модели.
        - Сохранение лучшей модели.
        """
        self.log.info("Запуск процесса обучения модели...")

        # 1. Определяем, какой файл с данными нам нужен
        cache_filename = get_cache_filename(experiment_cfg, self.cfg.PREPROCESSING_VERSION)
        cache_path = self.cfg.DATA_DIR / cache_filename

        # ПРИМЕЧАНИЕ: Логика self.data_saver.load() должна будет
        # научиться определять нужный файл на основе experiment_cfg.
        # Пока оставляем как есть.
        datasets = self.data_saver.load(file_path=cache_path)
        
        # --- ЗАГЛУШКА: Эмулируем результаты обучения ---
        import random
        ml_metrics = {"accuracy": random.uniform(0.5, 0.6), "loss": random.uniform(0.8, 1.2)}
        # --- КОНЕЦ ЗАГЛУШКИ ---
        self.log.info("Процесс обучения (заглушка) завершен.")
        return ml_metrics