# src/dataset_builder.py

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple, Generator

from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class DatasetBuilder:
    """
    Отвечает за преобразование плоских numpy-массивов
    в форматы, специфичные для разных типов моделей.
    """
    def __init__(self, cfg:AppConfig, log:AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def _sequence_generator(self, data:np.ndarray, target_indices:list[int], x_len: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Генератор, который "на лету" выдает по одному окну (X, y).
        """
        # x_len = self.cfg.X_LEN
        feature_indices = [i for i in range(data.shape[1]) if i not in target_indices]

        for i in range(len(data) - x_len):
            window = data[i : (i + x_len)]
            target = data[i + x_len]

            x_sample = window[:, feature_indices]
            y_sample = target[target_indices]

            yield x_sample.astype(np.float32), y_sample.astype(np.float32)

    def build(self, model_type:str, datasets:dict, target_cols:list[str], all_cols:list[str], experiment_cfg: ExperimentConfig) -> dict:
        """
        Главный метод-оркестратор. Вызывает нужный приватный build-метод
        на основе типа модели.

        Args:
            model_type (str): Тип модели ('lightgbm', 'keras', 'autots').
            datasets (dict): Словарь с numpy-массивами {'train': ..., 'val': ..., 'test': ...}.
            target_cols (list[str]): Список имен целевых колонок.
            all_cols (list[str]): Полный список всех колонок.
            experiment_cfg (ExperimentConfig): Конфигурация для доступа к x_len.

        Returns:
            dict: Словарь с подготовленными данными для модели.
        """
        x_len = experiment_cfg.common_params.get("x_len", 22)

        if model_type in ['lightgbm', 'catboost']: # Для всех табличных моделей
            return self._build_for_tabular(datasets, target_cols, all_cols)

        elif model_type in ['lstm', 'lstm_v2', 'af_lstm', 'tcn', 'transformer']: # Для всех Keras-моделей
            return self._build_for_keras(datasets, target_cols, all_cols, x_len)
        ###elif model_type in ['af_lstm']: # Для af-модели
        ###    return self._build_for_keras(datasets, target_cols, all_cols)
        elif model_type == 'autots':
            return self._build_for_autots(datasets, target_cols, all_cols)
        else:
            raise NotImplementedError(f"Логика построения датасета для типа модели '{model_type}' не реализована.")

    def _build_for_autots(self, datasets: dict, target_cols: list[str], all_cols: list[str]) -> dict:
        """
        Готовит данные для AutoTS. Возвращает pandas DataFrame.

        Args:
            datasets (dict): Словарь с numpy-массивами.
            target_cols (list[str]): Список имен целевых колонок.
            all_cols (list[str]): Полный список всех колонок.

        Returns:
            dict: Словарь с pandas DataFrame для каждой выборки.
        """
        self.log.info("Подготовка DataFrame'ов для модели AutoTS...")
        output = {}
        for key, data in datasets.items():
            # AutoTS предпочитает работать с pandas DataFrame
            output[f'{key}_df'] = pd.DataFrame(data, columns=all_cols)
        
        return output

    def _build_for_tabular(self, datasets: dict, target_cols: list[str], all_cols: list[str]) -> dict:
        """
        Готовит "плоские" данные для табличных моделей (LightGBM, CatBoost).

        Args:
            datasets (dict): Словарь с numpy-массивами {'train': ..., 'val': ..., 'test': ...}.
            target_cols (list[str]): Список имен целевых колонок.
            all_cols (list[str]): Полный список всех колонок.

        Returns:
            dict: Словарь с numpy-массивами X и y для каждой выборки.
        """
        self.log.info("Подготовка 'плоских' данных для табличной модели...")
        output = {}

        target_indices = [all_cols.index(c) for c in target_cols]
        feature_indices = [i for i in range(datasets['train'].shape[1]) if i not in target_indices]

        for key, data in datasets.items():
            # Для табличных моделей окна не нужны, просто разделяем X и y
            X_data = data[:, feature_indices]
            y_data = data[:, target_indices]
            
            # Сохраняем и как numpy-массивы (для обратной совместимости)
            output[f'X_{key}'] = X_data
            output[f'y_{key}'] = y_data

            # Сохраняем и как DataFrame с именами колонок
            feature_names = [all_cols[i] for i in feature_indices]
            output[f'X_{key}_df'] = pd.DataFrame(X_data, columns=feature_names)
            output[f'y_{key}_df'] = pd.DataFrame(y_data, columns=target_cols)

        return output
    
    def _build_for_keras(self, datasets: dict, target_cols: list[str], all_cols: list[str], x_len: int) -> dict:
        """
        Создает `tf.data.Dataset` с окнами для Keras-моделей.

        Args:
            datasets (dict): Словарь с numpy-массивами.
            target_cols (list[str]): Список имен целевых колонок.
            all_cols (list[str]): Полный список всех колонок.
            x_len (int): Длина окна последовательности.

        Returns:
            dict: Словарь, содержащий `tf.data.Dataset` для train/val и "плоские" X/y для test.
        """
        self.log.info("Подготовка оконных данных для Keras-модели...")
        output = {}
        target_indices = [all_cols.index(col) for col in target_cols]
        num_features = len(all_cols) - len(target_cols)

        for key, data in datasets.items():
            if len(data) > x_len:

                output_signature = (
                    tf.TensorSpec((x_len, num_features), tf.float32),
                    # tf.TensorSpec((self.cfg.X_LEN, num_features), tf.float32),
                    tf.TensorSpec((len(target_indices),), tf.float32)
                )
                dataset = tf.data.Dataset.from_generator(
                    lambda d=data: self._sequence_generator(d, target_indices, x_len),
                    output_signature=output_signature
                )

                # Ключи для словаря output. Для 'validation' используем 'val'
                output_key = 'val' if key == 'validation' else key

                # Делаем датасет многоразовым для обучения в течение нескольких эпох
                output[f'{output_key}_dataset'] = dataset.batch(self.cfg.BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

                X_flat, y_flat = self._create_sequences(data, target_indices, x_len)
                output[f'X_{output_key}'] = X_flat
                output[f'y_{output_key}'] = y_flat

            else:
                self.log.warning(f"Выборка '{key}' слишком мала для нарезки на окна. Пропускается.")

        # Рассчитываем правильное количество шагов для датасета на основе реального числа окон
        if 'train' in datasets and len(datasets['train']) > x_len:
            train_samples = len(datasets['train']) - x_len
            output['steps_per_epoch'] = int(np.ceil(train_samples / self.cfg.BATCH_SIZE))

        if 'validation' in datasets and len(datasets['validation']) > x_len:
            val_samples = len(datasets['validation']) - x_len
            output['validation_steps'] = int(np.ceil(val_samples / self.cfg.BATCH_SIZE))

        return output

    def _create_sequences(self, data:np.ndarray, target_indices:list[int], x_len: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Нарезает данные на последовательности (окна) и возвращает их как numpy-массивы.

        Args:
            data (np.ndarray): Входной массив данных (признаки + цели).
            target_indices (list[int]): Индексы колонок, которые являются целевыми.
            x_len (int): Длина окна последовательности.

        Returns:
            tuple[np.ndarray, np.ndarray]: Кортеж (X, y), где X - последовательности, y - метки.
        """
        feature_indices = [i for i in range(data.shape[1]) if i not in target_indices]
        
        X, y = [], []
        for i in range(len(data) - x_len):
            window = data[i : (i + x_len)]
            target = data[i + x_len]
            
            X.append(window[:, feature_indices])
            y.append(target[target_indices])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)