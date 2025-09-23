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
    # Общие параметры, не зависящие от конкретного trial
    common_params: dict
    # Гиперпараметры для конкретного trial
    model_params: dict = field(default_factory=dict)
    train_params: dict = field(default_factory=dict)
    # Параметры, специфичные для запуска
    log_history_per_epoch: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Преобразует конфиг в словарь для логирования в MLflow."""
        # Распаковываем общие параметры и объединяем с остальными
        params = self.common_params.copy()
        params.update(self.model_params)
        params.update(self.train_params)
        params["log_history_per_epoch"] = self.log_history_per_epoch
        return params