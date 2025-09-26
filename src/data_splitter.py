# src.data_splitter.py

from typing import Tuple, List, Literal
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

    def split(self, df: pd.DataFrame, mode: str, experiment_cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        """
        Выполняет хронологическое разделение данных в зависимости от режима.
        - В режиме 'search': делит на train, val, test.
        - В режиме 'train' может объединять train и val выборки,
        если в конфиге указан флаг 'train_on_full_data'.

                Args:
            df (pd.DataFrame): Полный DataFrame для разделения.
            mode (str): Режим работы ('search', 'train', 'finetune').
            experiment_cfg (ExperimentConfig): Конфигурация для доступа к флагам.

        Returns:
            Кортеж из трех DataFrame: (train_df, val_df|None, test_df).
        """
        self.log.info(f"Начинаем хронологическое разделение датасета в режиме '{mode}'...")

        # --- Определение параметров разделения на основе конфига и режима ---
        train_on_full = experiment_cfg.common_params.get('train_on_full_data', False)
        model_type = experiment_cfg.common_params.get('model_type')
        
        # Модели, которым всегда нужна валидационная выборка для early stopping
        needs_validation = model_type in ['lstm', 'lstm_v2', 'af_lstm', 'tcn', 'transformer']

        # Применяем специальную логику только при соблюдении ВСЕХ условий
        if mode == 'train' and train_on_full and not needs_validation:
            self.log.info("Обнаружен флаг 'train_on_full_data' для табличной модели. Val-выборка будет пустой, gap будет уменьшен.")
            val_size = 0
            gap_size = self.cfg.GAP_SIZE // 2
        else:
            val_size = self.cfg.VAL_SIZE
            gap_size = self.cfg.GAP_SIZE
        
        ########### # Флаг train_on_full_data имеет смысл только в режиме 'train'
        ########### #train_on_full = False
        ########### if mode == 'train':
        ###########     # В режиме train мы смотрим на train_mode_config
        ###########     config_block = experiment_cfg.base_config.get('train_mode', {})
        ###########     train_on_full = config_block.get('train_on_full_data', False)
        ########### 
        ########### if train_on_full and mode == 'train':
        ###########     self.log.info("Обнаружен флаг 'train_on_full_data'. Val-выборка будет пустой, gap будет уменьшен.")
        ###########     val_size = 0
        ###########     gap_size = self.cfg.GAP_SIZE // 2
        ########### else:
        ###########     val_size = self.cfg.VAL_SIZE
        ###########     gap_size = self.cfg.GAP_SIZE

        test_size = self.cfg.TEST_SIZE

        min_len = test_size + val_size + 2 * gap_size
        if len(df) < min_len:
            raise ValueError("Недостаточно данных для выполнения разделения с заданными параметрами.")

        test_df = df.iloc[-test_size:].copy()

        val_end_idx = -test_size - gap_size
        val_start_idx = val_end_idx - val_size
        val_df = df.iloc[val_start_idx:val_end_idx].copy() if val_size > 0 else None
                
        train_end_idx = val_start_idx - gap_size
        train_df = df.iloc[:train_end_idx].copy()
        
        val_shape = val_df.shape if val_df is not None else "None"
        self.log.info(f"Размеры выборок: Train={train_df.shape}, Validation={val_shape}, Test={test_df.shape}")
        return train_df, val_df, test_df

    def scale(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None, test_df: pd.DataFrame, experiment_cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, StandardScaler | None]:
        """Масштабирует переданные DataFrame с помощью StandardScaler."""
        self.log.info("Начинаем масштабирование признаков...")

        # Выбираем ВСЕ числовые колонки для обучения скейлера и масштабирования
        cols_to_scale = train_df.select_dtypes(include=np.number).columns.tolist()
        
        if not cols_to_scale:
            self.log.warning("Не найдено числовых колонок для масштабирования. Пропускаем шаг.")
            return train_df, val_df, test_df, None

        self.log.info(f"Количество колонок для масштабирования: {len(cols_to_scale)}")

        scaler = StandardScaler()

        train_df_scaled = train_df.copy()
        # Проверяем, существует ли val_df, прежде чем его копировать
        val_df_scaled = val_df.copy() if val_df is not None else None
        test_df_scaled = test_df.copy()
        
        # Приводим типы к float64
        train_df_scaled[cols_to_scale] = train_df_scaled[cols_to_scale].astype(np.float64)
        if val_df_scaled is not None:
            val_df_scaled[cols_to_scale] = val_df_scaled[cols_to_scale].astype(np.float64)
        test_df_scaled[cols_to_scale] = test_df_scaled[cols_to_scale].astype(np.float64)
        
        # Обучаем скейлер на ВСЕХ числовых колонках из train
        scaler.fit(train_df_scaled[cols_to_scale])
        ### scaler.fit(train_df_scaled[feature_cols])

        # Применяем скейлер ко всем выборкам
        train_df_scaled.loc[:, cols_to_scale] = scaler.transform(train_df_scaled[cols_to_scale])
        if val_df_scaled is not None:
            val_df_scaled.loc[:, cols_to_scale] = scaler.transform(val_df_scaled[cols_to_scale])
        test_df_scaled.loc[:, cols_to_scale] = scaler.transform(test_df_scaled[cols_to_scale])

        self.log.info("Масштабирование признаков завершено.")
        return train_df_scaled, val_df_scaled, test_df_scaled, scaler

    def _get_target_columns(self, experiment_cfg: ExperimentConfig) -> List[str]:
        """
        Определяет имена целевых столбцов на основе common_params.
        - Для classification: возвращает ['target'].
        - Для regression: возвращает список вида ['target_<name>'] для каждой
          колонки из common_params['targets'].
        """
        task_type = experiment_cfg.common_params.get("task_type")
        
        if task_type == "classification":
            return ['target']

        elif task_type == "regression":
            targets = experiment_cfg.common_params.get("targets")
            if not targets or not isinstance(targets, list):
                raise ValueError("Ключ 'targets' (список колонок) не найден или имеет неверный формат в common_params.")
            # Возвращаем имена целевых колонок с добавленным префиксом 'target_'
            return [f"target_{col}" for col in targets]
        else:
            raise NotImplementedError(f"Логика для определения целевых столбцов типа '{task_type}' не реализована.")