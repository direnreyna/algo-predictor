# src/app_logger.py

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from .app_config import AppConfig

class AppLogger:
    """
    Класс для централизованного логирования. Использует паттерн Singleton.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file_path: Optional[Path] = None):
        if self._initialized:
            return
        
        # Если путь не передан явно, берем его из конфига
        if log_file_path is None:

            # Создаем (или получаем существующий) экземпляр AppConfig
            cfg = AppConfig()
            # Обращаемся к атрибуту экземпляра
            log_file_path = cfg.LOGS_DIR / "predictor.log"

        self.logger = logging.getLogger('algo_logger')
        self.logger.setLevel(logging.INFO)

        # Избегаем дублирования обработчиков
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        
        log_dir = log_file_path.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        # Обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Обработчик для файла
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._initialized = True

    def info(self, message: str, **kwargs):
        """Логирует информационное сообщение."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Логирует предупреждение."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Логирует сообщение об ошибке."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Логирует критическое сообщение."""
        self.logger.critical(message, **kwargs)

    def separator(self):
        """Логирует разделитель."""
        self.logger.info("=" * 80)