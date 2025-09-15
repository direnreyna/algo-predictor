# src/model_factory.py

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """
    Базовый класс для всех наших моделей.
    Он гарантирует, что любая модель будет иметь одинаковый "интерфейс".
    """
    def __init__(self, model_params:dict):
        self.model = None
        self.params = model_params

    @abstractmethod
    def train(self, data_dict: dict) -> dict:
        """
        Метод для обучения модели.

        Args:
            data_dict (dict): Словарь с подготовленными данными,
                              например {'X_train': ..., 'y_train': ...}.

        Returns:
            dict: Словарь с историей обучения (например, {'loss': [...]}).
        """
        pass

    @abstractmethod
    def predict(self, X) -> Any:
        """Метод для получения предсказаний."""
        pass

    @abstractmethod
    def save(self, path:Path) -> None:
        """Метод для сохранения модели."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path:Path) -> 'BaseModel':
        """Метод для загрузки модели."""
        pass

class ModelFactory:
    """Это "фабрика" моделей, которая будет создавать нужную модель по имени."""
    @staticmethod
    def get_model(model_type:str, model_params:dict|None=None):
        if model_params is None:
            model_params = {}
    
        if model_type == "lightgbm":
            from .models.lgbm_model import LightGBMModel
            return LightGBMModel(model_params)
        
        elif model_type == "lstm":
            from .models.keras_lstm_model import KerasLSTMModel
            return KerasLSTMModel(model_params)
        
        ################################################################################
        # Здесь будут другие модели: 'lstm', 'autots' и т.д.
        ################################################################################        
        
        else:
            raise ValueError(f"Модель типа '{model_type}' не поддерживается.")