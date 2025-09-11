# src/config_loader.py

import yaml
from pathlib import Path
from .entities import ExperimentConfig

class ConfigLoader:
    """
    Отвечает за загрузку, парсинг и валидацию конфигурационных
    файлов для экспериментов.
    """
    @staticmethod
    def load_from_yaml(config_path: Path) -> dict:
        """Загружает базовый словарь из YAML-файла."""
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        return base_config

    @staticmethod
    def create_experiment_config(base_config: dict) -> ExperimentConfig:
        """Создает экземпляр ExperimentConfig из 'статичной' части конфига."""
        # Убираем секцию search_space, так как она не является частью паспорта
        static_params = {k: v for k, v in base_config.items() if k != 'search_space'}
        return ExperimentConfig(**static_params)