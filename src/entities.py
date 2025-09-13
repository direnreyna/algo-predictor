# src/entities.py

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Неизменяемый dataclass, который служит "паспортом" для одного эксперимента.
    frozen=True делает экземпляр неизменяемым после создания, что предотвращает
    случайные ошибки.
    """
    # 1. Параметры предобработки
    asset_name: str
    feature_set_name: str
    labeling_horizon: int
    task_type: str
    
    # 2. Параметры модели
    model_type: str
    
    # Опциональные параметры с дефолтными значениями
    column_mapping: dict | None = None
    model_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует конфиг в словарь для логирования."""
        return {
            "asset_name": self.asset_name,
            "feature_set_name": self.feature_set_name,
            "labeling_horizon": self.labeling_horizon,
            "task_type": self.task_type,
            "model_type": self.model_type,
            **self.model_params # Распаковываем параметры модели в основной словарь
        }