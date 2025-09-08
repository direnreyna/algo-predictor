# src.data_preparer.py

import pandas as pd
import numpy as np

from .app_config import AppConfig
from .app_logger import AppLogger
from .feature_engineer import FeatureEngineer
from .data_splitter import DataSplitter
from .data_labeler import DataLabeler
from .data_saver import DataSaver

class DataPreparer:
    """
    Оркестратор предобработки данных.
    Отвечает за полный цикл подготовки датасета для обучения.
    Этапы:
    1. Расчет технических признаков:
       - Генерация индикаторов (RSI, MACD и т.д.) на основе OHLCV.
    2. Обогащение альтернативными данными (Data Enrichment):
       - Интеграция внешних источников (новости, соцсети и т.д.).
    3. Создание целевой переменной (Labeling):
       - Формирование классов (UP/DOWN/SIDEWAYS) или значений для регрессии.
    4. Разделение на train/val/test с учетом разрывов (gaps).
    5. Нормализация признаков (Scaling).
    6. Нарезка на окна (Sequencing).
    7. Кеширование (Caching):
       - Сохранение готовых выборок в файл (.npz) для быстрого повторного использования.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.feature_engineer = FeatureEngineer(cfg, log)
        self.data_labeler = DataLabeler(cfg, log)
        self.data_splitter = DataSplitter(cfg, log)
        self.data_saver = DataSaver(cfg, log)
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df: pd.DataFrame) -> None: # Возвращать будет несколько выборок
        """
        Выполняет полный цикл предобработки данных.
        
        :param df: Исходный DataFrame для обработки.
        :return: Кортеж с подготовленными выборками (например, X_train, y_train, X_val, y_val, ...).
        """
        self.log.info("Начало процесса предобработки данных")
        
        # 0. ПРОПУСКАЕМ, если данные уже обработаны и сохранены на диск в формате .npz
        if self.cfg.PREPARED_DATA_PATH.exists():
            self.log.info(f"Найден готовый файл с данными '{self.cfg.PREPARED_DATA_PATH.name}'. Пропускаем генерацию.")
            return

        self.log.info("Готовые данные не найдены. Запуск полного цикла предобработки.")
        
        # 1. Обогащение признаками (Feature Engineering)
        df_with_features = self.feature_engineer.run(df)
        
        # 2. Обогащение доп. данными (пока пропуск)
        
        # 3. Создание целевой переменной
        df_labeled = self.data_labeler.run(df_with_features)
        
        # 4. Разделение на выборки и Нормализация
        train_df, val_df, test_df = self.data_splitter.run(df_labeled)
        
        # 5. Сохранение выборок
        self.data_saver.save(train=train_df, validation=val_df, test=test_df)
        
        self.log.info("Процесс предобработки данных завершен.")