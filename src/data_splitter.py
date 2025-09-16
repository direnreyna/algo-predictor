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

    ###def run(self, df:pd.DataFrame, experiment_cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    ###    """
    ###    Разделяет данные и, опционально, масштабирует их.
    ###
    ###    Args:
    ###        df (pd.DataFrame): Полный DataFrame. MESSAGE: The user has run the code with differentiation disabled, but the `KeyError: '
    ###        experiment_cfg (ExperimentConfig): Конфигурация эксперимента.
    ###        scale_data (bool): Если True, выполняет масштабирование.
    ###
    ###    Returns:
    ###        Кортеж из (train_df, val_df, test_df, scalerX_val'` persists. The log clearly shows `WARNING - Выборка 'validation' слишком мала для нарез | None).
    ###    """
    ###    train_df, val_df, test_df = self.split(df)
    ###    return self.scale(train_df, val_df, test_df, experiment_cfg)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Выполняет только хронологическое разделение данных.

        Args:
            df (pd.DataFrame): Полный DataFrame для разделения.

        Returns:
            Кортеж из трех DataFrame: (train_df, val_df, test_df).
        """
        self.log.info("Начинаем хронологическое разделение датасета...")

        if len(df) < self.cfg.TEST_SIZE + self.cfg.VAL_SIZE + 2 * self.cfg.GAP_SIZE:
            raise ValueError("Недостаточно данных для выполнения разделения с заданными параметрами.")

        test_df = df.iloc[-self.cfg.TEST_SIZE:].copy()
        
        val_end_idx = -self.cfg.TEST_SIZE - self.cfg.GAP_SIZE
        val_start_idx = val_end_idx - self.cfg.VAL_SIZE
        val_df = df.iloc[val_start_idx:val_end_idx].copy()
        
        train_end_idx = val_start_idx - self.cfg.GAP_SIZE
        train_df = df.iloc[:train_end_idx].copy()
        
        self.log.info(f"Размеры выборок после разделения: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")
        return train_df, val_df, test_df

    def scale(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, experiment_cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | None]:
        """Масштабирует переданные DataFrame с помощью StandardScaler."""
        self.log.info("Начинаем масштабирование признаков...")

        ### target_cols = self._get_target_columns(experiment_cfg)
        ### numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
        ### feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        # Выбираем ВСЕ числовые колонки для обучения скейлера и масштабирования
        cols_to_scale = train_df.select_dtypes(include=np.number).columns.tolist()
        
        if not cols_to_scale:
            self.log.warning("Не найдено числовых колонок для масштабирования. Пропускаем шаг.")
            return train_df, val_df, test_df, None

        self.log.info(f"Количество колонок для масштабирования: {len(cols_to_scale)}")

        scaler = StandardScaler()

        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()
        
        # Приводим типы к float64
        train_df_scaled[cols_to_scale] = train_df_scaled[cols_to_scale].astype(np.float64)
        val_df_scaled[cols_to_scale] = val_df_scaled[cols_to_scale].astype(np.float64)
        test_df_scaled[cols_to_scale] = test_df_scaled[cols_to_scale].astype(np.float64)

        ### train_df_scaled[feature_cols] = train_df_scaled[feature_cols].astype(np.float64)
        ### val_df_scaled[feature_cols] = val_df_scaled[feature_cols].astype(np.float64)
        ### test_df_scaled[feature_cols] = test_df_scaled[feature_cols].astype(np.float64)
        
        # Обучаем скейлер на ВСЕХ числовых колонках из train
        scaler.fit(train_df_scaled[cols_to_scale])
        ### scaler.fit(train_df_scaled[feature_cols])

        # Применяем скейлер ко всем выборкам
        train_df_scaled.loc[:, cols_to_scale] = scaler.transform(train_df_scaled[cols_to_scale])
        val_df_scaled.loc[:, cols_to_scale] = scaler.transform(val_df_scaled[cols_to_scale])
        test_df_scaled.loc[:, cols_to_scale] = scaler.transform(test_df_scaled[cols_to_scale])

        ### train_df_scaled.loc[:, feature_cols] = scaler.transform(train_df_scaled[feature_cols])
        ### val_df_scaled.loc[:, feature_cols] = scaler.transform(val_df_scaled[feature_cols])
        ### test_df_scaled.loc[:, feature_cols] = scaler.transform(test_df_scaled[feature_cols])

        self.log.info("Масштабирование признаков завершено.")
        return train_df_scaled, val_df_scaled, test_df_scaled, scaler

    def _get_target_columns(self, experiment_cfg: ExperimentConfig) -> List[str]:
        """Определяет имена целевых столбцов на основе типа задачи."""
        if experiment_cfg.task_type == "classification":
            return ['target']
        elif experiment_cfg.task_type == "regression":
            return ['target_high', 'target_low']
        else:
            # На случай добавления новых типов задач
            raise NotImplementedError(f"Логика для определения целевых столбцов типа '{experiment_cfg.task_type}' не реализована.")