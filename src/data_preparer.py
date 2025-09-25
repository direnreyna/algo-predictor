# src.data_preparer.py

import pandas as pd
import numpy as np
import json
import joblib
import mlflow

from statsmodels.tsa.stattools import adfuller

from .app_config import AppConfig
from .app_logger import AppLogger
from .feature_engineer import FeatureEngineer
from .data_splitter import DataSplitter
from .data_labeler import DataLabeler
from .data_saver import DataSaver
from .entities import ExperimentConfig
from .file_loader import FileLoader
from .statistical_analyzer import StatisticalAnalyzer
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
        self.statistical_analyzer = StatisticalAnalyzer(cfg, log)
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

        asset_name = experiment_cfg.common_params.get("asset_name")
        if not asset_name:
            raise ValueError("Ключ 'asset_name' не найден в common_params.")
        file_name = f"{asset_name}.csv"
        df = self.file_loader.read_csv(file_name, experiment_cfg=experiment_cfg)

        # Статистический анализ
        global_insights = self.statistical_analyzer.analyze(df, experiment_cfg=experiment_cfg)
        if mlflow.active_run():
            mlflow.log_params(global_insights)

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
        
        # ### # --- Диагностика: Проверяем DataFrame ПОСЛЕ Feature Engineering и Labeling ---
        # self.log.info(f"--- ДИАГНОСТИКА: DataFrame ПОСЛЕ Feature Engineering и Labeling ---")
        # self.log.info(f"Shape: {df_labeled.shape}\n{df_labeled.head().to_string()}")
        # ### # --- Конец Диагностики ---

        # 6. Разделение на выборки (ДО трансформаций и масштабирования)
        train_df, val_df, test_df = self.data_splitter.split(df_labeled)
        
        # Сохранение копии test_df до всех трансформаций. Нужно для восстановления мастштабированных приращений цен (без самих цен)
        original_test_df = test_df.copy()

        # 7. Статистическая трансформация (ДО масштабирования)
        self.log.info("Запуск фазы 2: Статистическая трансформация данных...")
        self.statistical_analyzer.fit(train_df, global_insights)

        train_df, was_differenced_flag = self.statistical_analyzer.transform(train_df, experiment_cfg=experiment_cfg)
        val_df, _ = self.statistical_analyzer.transform(val_df, experiment_cfg=experiment_cfg)
        test_df, _ = self.statistical_analyzer.transform(test_df, experiment_cfg=experiment_cfg)
        
        # Устанавливаем флаг в конфиге на основе результата трансформации train-выборки
        experiment_cfg.was_differenced = was_differenced_flag
        self.log.info("Статистическая трансформация завершена.")

        # ### # --- Диагностика: Проверяем Train DataFrame ПОСЛЕ статистической трансформации
        # self.log.info(f"--- ДИАГНОСТИКА: Train DataFrame ПОСЛЕ статистической трансформации ---")
        # self.log.info(f"Shape: {train_df.shape}\n{train_df.head().to_string()}")
        # self.log.info(f"Статистика (describe):\n{train_df.describe().to_string()}")
        # ### # --- Конец Диагностики ---

        ### # 8. Контрольная проверка стационарности после всех манипуляций
        ### diff_mode = experiment_cfg.common_params.get("differencing", "false")
        ### was_differenced = (diff_mode == "true") or (diff_mode == "auto" and not global_insights.get("is_stationary", True))
        ### experiment_cfg.was_differenced = was_differenced
        ### 
        ### if was_differenced:
        ###     self.log.info("Контрольная проверка стационарности ряда 'Close' ПОСЛЕ трансформации...")
        ###     adf_result_after = adfuller(train_df['Close'].dropna())
        ###     p_value_after = adf_result_after[1]
        ###     is_stationary_after = p_value_after < 0.05
        ###     log_func = self.log.info if is_stationary_after else self.log.warning
        ###     log_func(f"  ADF-тест ПОСЛЕ (p-value): {p_value_after:.4f}. Стационарный: {is_stationary_after}")
        ###     if mlflow.active_run():
        ###         mlflow.log_metric("adf_pvalue_after_transform", p_value_after)

        # 8. Масштабирование (ПОСЛЕ трансформаций)
        self.log.info("Масштабирование трансформированных данных...")
        train_df, val_df, test_df, scaler = self.data_splitter.scale(
            train_df, val_df, test_df, experiment_cfg
        )
        self.log.info("Масштабирование завершено.")
        
        # ### # --- Диагностика: Проверяем Train DataFrame ПОСЛЕ масштабирования
        # self.log.info(f"--- ДИАГНОСТИКА: Train DataFrame ПОСЛЕ масштабирования ---")
        # self.log.info(f"Shape: {train_df.shape}\n{train_df.head().to_string()}")
        # self.log.info(f"Статистика (describe):\n{train_df.describe().to_string()}")
        # ### # --- Конец Диагностики ---

        # 9. Сохранение скейлера
        scaler_path = cache_path.with_suffix('.joblib')
        try:
            joblib.dump(scaler, scaler_path)
            self.log.info(f"Скейлер успешно сохранен в '{scaler_path.name}'")
        except Exception as e:
            self.log.error(f"Ошибка при сохранении скейлера: {e}")
            raise

        # 10. Сохраняем метаданные (имена колонок)
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

        # 11. Сохранение выборок
        self.data_saver.save(file_path=cache_path, train=train_df, validation=val_df, test=test_df, original_test=original_test_df)
        
        self.log.info("Процесс предобработки данных завершен.")