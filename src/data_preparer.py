# src.data_preparer.py

import pandas as pd
import numpy as np
import json
import joblib

from .app_config import AppConfig
from .app_logger import AppLogger
from .feature_engineer import FeatureEngineer
from .data_splitter import DataSplitter
from .data_labeler import DataLabeler
from .data_saver import DataSaver
from .entities import ExperimentConfig
from .file_loader import FileLoader
from .cache_utils import get_cache_filename

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
        self.file_loader = FileLoader(cfg, log)
        self.feature_engineer = FeatureEngineer(cfg, log)
        self.data_labeler = DataLabeler(cfg, log)
        self.data_splitter = DataSplitter(cfg, log)
        self.data_saver = DataSaver(cfg, log)
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, experiment_cfg: ExperimentConfig) -> None:
        """
        Выполняет полный цикл предобработки данных.
        
        :param df: Исходный DataFrame для обработки.
        :return: Кортеж с подготовленными выборками (например, X_train, y_train, X_val, y_val, ...).
        """
        self.log.info("Начало процесса предобработки данных")
        
        # 0. Генерируем имя файла и проверяем кеш
        cache_filename = get_cache_filename(experiment_cfg, self.cfg.PREPROCESSING_VERSION)
        cache_path = self.cfg.DATA_DIR / cache_filename

        # 1. ПРОПУСКАЕМ, если данные уже обработаны и сохранены на диск в формате .npz
        if cache_path.exists():
            self.log.info(f"Найден готовый файл с данными '{cache_filename}'. Пропускаем генерацию.")
            return
        self.log.info("Готовые данные не найдены. Запуск полного цикла предобработки.")
        
        # 2: Загрузка данных
        file_name = f"{experiment_cfg.asset_name}.csv" 
        df = self.file_loader.read_csv(file_name, experiment_cfg=experiment_cfg)

        ### # --- Диагностика: Проверяем NaN после загрузки ---
        ### initial_nan_count = df.isna().sum().sum()
        ### if initial_nan_count > 0:
        ###     self.log.warning(f"В исходных данных после загрузки обнаружено {initial_nan_count} NaN.")
        ### else:
        ###     self.log.info("Проверка исходных данных: NaN не обнаружены.")
        ### # --- Конец Диагностики ---

        # 3. Обогащение признаками (Feature Engineering)
        df_with_features = self.feature_engineer.run(df, experiment_cfg)

        # 4. Обогащение доп. данными (пока пропуск)
        
        # 5. Создание целевой переменной
        df_labeled = self.data_labeler.run(df_with_features, experiment_cfg)
        
        # 6. Разделение на выборки и Нормализаци
        train_df, val_df, test_df, scaler = self.data_splitter.run(df_labeled, experiment_cfg)

        # 7. Сохранение скейлера
        scaler_path = cache_path.with_suffix('.joblib')
        try:
            joblib.dump(scaler, scaler_path)
            self.log.info(f"Скейлер успешно сохранен в '{scaler_path.name}'")
        except Exception as e:
            self.log.error(f"Ошибка при сохранении скейлера: {e}")
            raise

        # 8. Сохраняем метаданные (имена колонок)
        metadata = {
            'columns': train_df.columns.tolist(),
            'target_columns': self.data_splitter._get_target_columns(experiment_cfg)
        }
        metadata_path = cache_path.with_suffix('.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.log.info(f"Метаданные успешно сохранены в '{metadata_path.name}'")
        except Exception as e:
            self.log.error(f"Ошибка при сохранении метаданных: {e}")
            raise

        ### # Диагностика перед сохранением выборок
        ### self.log.info(f"--- ДИАГНОСТИКА: Данные перед сохранением выборок (Train) ---")
        ### train_df.info()
        ### self.log.info(f"Первые 5 строк:\n{train_df.head().to_string()}")
        ### # --- Конец диагностики ---

        # 9. Сохранение выборок
        self.data_saver.save(file_path=cache_path, train=train_df, validation=val_df, test=test_df)
        
        self.log.info("Процесс предобработки данных завершен.")