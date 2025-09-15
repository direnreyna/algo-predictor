# src/metrics_calculator.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MetricsCalculator:
    """
    Отвечает за расчет ML-метрик для задач регрессии и классификации.
    """
    @staticmethod
    def calculate(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Рассчитывает и возвращает словарь с соответствующими метриками.

        Args:
            task_type (str): Тип задачи ('regression' или 'classification').
            y_true (np.ndarray): Истинные значения.
            y_pred (np.ndarray): Предсказанные значения.

        Returns:
            dict: Словарь с рассчитанными метриками.
        """
        if task_type == "regression":
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
        elif task_type == "classification":
            # Для классификации предполагаем, что y_pred - это argmax от logits
            y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            return {
                'accuracy': accuracy_score(y_true_labels, y_pred_labels),
                'f1_score_macro': f1_score(y_true_labels, y_pred_labels, average='macro'),
                'precision_macro': precision_score(y_true_labels, y_pred_labels, average='macro'),
                'recall_macro': recall_score(y_true_labels, y_pred_labels, average='macro')
            }
        else:
            raise ValueError(f"Неизвестный тип задачи для расчета метрик: {task_type}")