# src.data_splitter.py

from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class DataSplitter:
    """
    Отвечает за разделение данных на выборки и их масштабирование.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df:pd.DataFrame, experiment_cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]: ##ДОБАВЛЕН БЛОК METHOD
        """
        Разделяет и масштабирует данные на train, validation и test.

        :param df: Полный DataFrame с признаками и метками.
        :return: Кортеж из четырех элементов: (train_df, val_df, test_df, scaler).
        """

        self.log.info("Начинаем разделение и нормирование датасета...")

        # 1. Хронологическое разделение с разрывами (gaps)
        # Убедимся, что данных достаточно для разделения
        if len(df) < self.cfg.TEST_SIZE + self.cfg.VAL_SIZE + 2 * self.cfg.GAP_SIZE:
            raise ValueError("Недостаточно данных для выполнения разделения с заданными параметрами.")

        test_df = df.iloc[-self.cfg.TEST_SIZE:].copy()
        
        val_end_idx = -self.cfg.TEST_SIZE - self.cfg.GAP_SIZE
        val_start_idx = val_end_idx - self.cfg.VAL_SIZE
        val_df = df.iloc[val_start_idx:val_end_idx].copy()
        
        train_end_idx = val_start_idx - self.cfg.GAP_SIZE
        train_df = df.iloc[:train_end_idx].copy()
        
        self.log.info(f"Размеры выборок: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")

        # 2. Определение признаков для масштабирования (все, кроме целевых)
        target_cols = self._get_target_columns(experiment_cfg)

        # Выбираем только числовые колонки для масштабирования, исключая целевые
        numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        self.log.info(f"Количество признаков для масштабирования: {len(feature_cols)}")

        # 3. Обучение скейлера ТОЛЬКО на train выборке
        scaler = StandardScaler()

        # Явно приводим типы к float64 перед масштабированием, чтобы избежать FutureWarning
        train_df[feature_cols] = train_df[feature_cols].astype(np.float64)
        val_df[feature_cols] = val_df[feature_cols].astype(np.float64)
        test_df[feature_cols] = test_df[feature_cols].astype(np.float64)

        scaler.fit(train_df[feature_cols])

        # 4. Применение скейлера ко всем выборкам
        # Используем .loc для присвоения, чтобы избежать SettingWithCopyWarning
        train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
        val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
        test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

        self.log.info("Масштабирование признаков завершено.")
        
        return train_df, val_df, test_df, scaler

    def _get_target_columns(self, experiment_cfg: ExperimentConfig) -> List[str]:
        """Определяет имена целевых столбцов на основе типа задачи."""
        if experiment_cfg.task_type == "classification":
            return ['target']
        elif experiment_cfg.task_type == "regression":
            return ['target_high', 'target_low']
        else:
            # На случай добавления новых типов задач
            raise NotImplementedError(f"Логика для определения целевых столбцов типа '{experiment_cfg.task_type}' не реализована.")