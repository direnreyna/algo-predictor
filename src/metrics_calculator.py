# src/metrics_calculator.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .entities import ExperimentConfig

class MetricsCalculator:
    """
    Отвечает за расчет ML-метрик для задач регрессии и классификации.
    """
    @staticmethod
    def calculate(
        task_type: str,
        y_true_abs: np.ndarray,
        y_pred_abs: np.ndarray
    ) -> dict:
        """
        Рассчитывает и возвращает словарь с соответствующими метриками.
        Для регрессии выполняет обратное преобразование для корректной оценки.

        Args:
            task_type (str): Тип задачи ('regression' или 'classification').
            y_true_abs (np.ndarray): Истинные абсолютные значения.
            y_pred_abs (np.ndarray): Предсказанные абсолютные значения.
                    
        Returns:
            dict: Словарь с рассчитанными метриками.
        """
        if task_type == "regression":

            ### # Извлекаем истинные АБСОЛЮТНЫЕ значения
            ### num_predictions = len(y_pred)
            ### offset = len(original_test_df) - num_predictions
            ### target_names = experiment_cfg.common_params.get("targets", [])
            ### 
            ### y_true_abs_list = []
            ### for name in target_names:
            ###     # Берем реальные будущие значения из оригинального датасета
            ###     true_values = original_test_df[name].iloc[offset:].values
            ###     y_true_abs_list.append(true_values)
            ### 
            ### y_true_abs = np.stack(y_true_abs_list, axis=1)
            ### 
            ### # Восстанавливаем предсказанные АБСОЛЮТНЫЕ значения
            ### y_pred_abs = y_pred
            ### 
            ### # Шаг 1: Обратное масштабирование (Inverse Scaling)
            ### unscaled_log_diffs = y_pred
            ### if scaler:
            ###     target_names = experiment_cfg.common_params.get("targets", [])
            ###     ### target_names = experiment_cfg.common_params.get("targets", [])
            ###     target_indices = [all_cols.index(f"target_{name}") for name in target_names]
            ### 
            ###     # Функция для обратного преобразования одного вектора (y_true или y_pred)
            ###     def inverse_transform_targets(targets_scaled: np.ndarray) -> np.ndarray:
            ###         temp_array = np.zeros((len(targets_scaled), len(all_cols)))
            ###         for i, idx in enumerate(target_indices):
            ###             temp_array[:, idx] = targets_scaled[:, i]
            ###         unscaled_array = scaler.inverse_transform(temp_array)
            ###         return unscaled_array[:, target_indices]
            ### 
            ###     unscaled_log_diffs = inverse_transform_targets(y_pred)
            ###     ### y_pred_abs = inverse_transform_targets(y_pred)
            ### 
            ### # Шаг 2: Обратное дифференцирование (Inverse Differencing) и экспоненцирование
            ### y_pred_abs = np.zeros_like(unscaled_log_diffs)
            ### if experiment_cfg.was_differenced:
            ### 
            ###     ### num_predictions = len(y_pred_abs)
            ###     ### offset = len(original_test_df) - num_predictions
            ###     
            ###     ### target_names = experiment_cfg.common_params.get("targets", [])
            ###     target_names = experiment_cfg.common_params.get("targets", [])
            ###    
            ###     # Восстанавливаем абсолютные значения для каждого таргета
            ###     for i, name in enumerate(target_names):
            ###         # Берем "точки отсчета" из ОРИГИНАЛЬНОГО датафрейма
            ###         ### last_known_values = original_test_df[name].iloc[offset-1 : -1]
            ###         last_known_log_values = np.log(original_test_df[name].iloc[offset-1 : -1].values + 1e-8)
            ### 
            ###         # Сначала восстанавливаем логарифм цены, потом экспоненцируем
            ###         predicted_log_prices = last_known_log_values + unscaled_log_diffs[:, i]
            ###         y_pred_abs[:, i] = np.exp(predicted_log_prices)
            ###         ### y_pred_abs[:, i] = last_known_values.values + y_pred_abs[:, i]
            ### else:
            ###     # Если дифференцирования не было, то unscaled_log_diffs - это log(price). Просто экспоненцируем.
            ###     y_pred_abs = np.exp(unscaled_log_diffs)

            return {
                'mse': mean_squared_error(y_true_abs, y_pred_abs),
                'mae': mean_absolute_error(y_true_abs, y_pred_abs),
                'r2_score': r2_score(y_true_abs, y_pred_abs)
            }
        elif task_type == "classification":
            # Для классификации предполагаем, что y_pred - это argmax от logits
            ### y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            ### y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            y_pred_labels = np.argmax(y_pred_abs, axis=1) if y_pred_abs.ndim > 1 else y_pred_abs
            y_true_labels = np.argmax(y_true_abs, axis=1) if y_true_abs.ndim > 1 else y_true_abs
            return {
                'accuracy': accuracy_score(y_true_labels, y_pred_labels),
                'f1_score_macro': f1_score(y_true_labels, y_pred_labels, average='macro'),
                'precision_macro': precision_score(y_true_labels, y_pred_labels, average='macro'),
                'recall_macro': recall_score(y_true_labels, y_pred_labels, average='macro')
            }
        else:
            raise ValueError(f"Неизвестный тип задачи для расчета метрик: {task_type}")