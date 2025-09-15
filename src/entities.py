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
 
    # 3. Параметры запуска/оркестрации
    ### finetune_from_run: dict | None = None
    log_history_per_epoch: bool = False
    
    # Опциональные параметры с дефолтными значениями
    column_mapping: dict | None = None
    ### model_params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    train_params: dict = field(default_factory=dict)
    finetune_from_run: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Преобразует конфиг в словарь для логирования."""
        params = {
            "asset_name": self.asset_name,
            "feature_set_name": self.feature_set_name,
            "labeling_horizon": self.labeling_horizon,
            "task_type": self.task_type,
            "model_type": self.model_type,
            "finetune_from_run_id": self.finetune_from_run.get("run_id") if self.finetune_from_run else None,
            "log_history_per_epoch": self.log_history_per_epoch
        }
        params.update(self.model_params)
        params.update(self.train_params)
        return params