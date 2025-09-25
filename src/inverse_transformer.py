# src/inverse_transformer.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

from .entities import ExperimentConfig
from .app_logger import AppLogger

class InverseTransformer:
    """
    Централизованный класс для выполнения обратного преобразования
    предсказаний модели к их исходному, абсолютному масштабу.

    Отвечает за корректную последовательность операций:
    1. Обратное масштабирование (inverse scaling).
    2. Обратное дифференцирование (inverse differencing) и экспоненцирование.

    Хранит все необходимые для этого артефакты: скейлер,
    оригинальный тестовый датафрейм (для точек отсчета) и метаданные.
    """

    def __init__(self,
                 scaler: StandardScaler,
                 original_test_df: pd.DataFrame,
                 all_cols: List[str],
                 experiment_cfg: ExperimentConfig):
        """
        Инициализирует трансформер.

        Args:
            scaler (StandardScaler): Обученный скейлер.
            original_test_df (pd.DataFrame): Нетронутый тестовый датафрейм,
                содержащий исходные цены.
            all_cols (List[str]): Полный список колонок в том порядке,
                в котором обучался скейлер.
            experiment_cfg (ExperimentConfig): Конфигурация эксперимента.
        """
        self.log = AppLogger()
        self.scaler = scaler
        self.original_test_df = original_test_df
        self.all_cols = all_cols
        self.experiment_cfg = experiment_cfg
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Выполняет полный цикл обратного преобразования предсказаний.

        Args:
            predictions (np.ndarray): Массив предсказаний от модели
                (масштабированных и, возможно, дифференцированных).

        Returns:
            np.ndarray: Массив предсказаний в их абсолютном значении (ценах).
        """
        if self.scaler is None:
            self.log.warning("Скейлер не предоставлен, обратное масштабирование пропускается.")
            # Если не было масштабирования, значит, не было и трансформаций
            return predictions

        # Шаг 1: Обратное масштабирование (Inverse Scaling)
        target_names_raw = self.experiment_cfg.common_params.get("targets", [])
        # Имена целевых колонок в обработанном датафрейме имеют префикс 'target_'
        target_cols_processed = [f"target_{name}" for name in target_names_raw]
        target_indices = [self.all_cols.index(col) for col in target_cols_processed]

        # Создаем "пустышку" для корректного inverse_transform
        temp_array = np.zeros((len(predictions), len(self.all_cols)))
        for i, idx in enumerate(target_indices):
            temp_array[:, idx] = predictions[:, i]

        unscaled_array = self.scaler.inverse_transform(temp_array)
        # Извлекаем только наши немасштабированные предсказания (это еще логарифмы или их разности)
        unscaled_predictions = unscaled_array[:, target_indices]

        # Шаг 2: Обратное дифференцирование и экспоненцирование
        absolute_predictions = np.zeros_like(unscaled_predictions)
        num_predictions = len(predictions)
        offset = len(self.original_test_df) - num_predictions

        if self.experiment_cfg.was_differenced:
            self.log.info("Выполняется обратное дифференцирование и экспоненцирование.")
            # Восстанавливаем абсолютные значения для каждого таргета
            for i, name in enumerate(target_names_raw):
                # Берем логарифм "точки отсчета" из ОРИГИНАЛЬНОГО датафрейма
                last_known_log_values = np.log(self.original_test_df[name].iloc[offset - 1: -1].values + 1e-8)

                # Сначала восстанавливаем логарифм цены, прибавляя предсказанное приращение
                predicted_log_prices = last_known_log_values + unscaled_predictions[:, i]
                # Затем экспоненцируем, чтобы получить абсолютную цену
                absolute_predictions[:, i] = np.exp(predicted_log_prices)
        else:
            self.log.info("Выполняется только экспоненцирование (дифференцирование не применялось).")
            # Если дифференцирования не было, то unscaled_predictions - это уже log(price)
            absolute_predictions = np.exp(unscaled_predictions)

        return absolute_predictions