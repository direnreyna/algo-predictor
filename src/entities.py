# src/entities.py

from dataclasses import dataclass, field
from typing import Any

@dataclass()
class ExperimentConfig:
    """
    Dataclass, который служит "паспортом" для одного эксперимента.
    """
    # Общие параметры, не зависящие от конкретного trial
    common_params: dict
    base_config: dict
    # Гиперпараметры для конкретного trial
    model_params: dict = field(default_factory=dict)
    train_params: dict = field(default_factory=dict)
    # Параметры, специфичные для запуска
    log_history_per_epoch: bool = False
    # Специальный флаг, изменяемый по ходу программы: если было применено дифференцирование колонок таргетов = True
    was_differenced: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Преобразует конфиг в словарь для логирования в MLflow."""
        # Распаковываем общие параметры и объединяем с остальными
        params = self.common_params.copy()
        params.update(self.model_params)
        params.update(self.train_params)
        params["log_history_per_epoch"] = self.log_history_per_epoch
        params["was_differenced"] = self.was_differenced
        return params