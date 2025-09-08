# src/downloader.py

import requests
from pathlib import Path
from .app_config import AppConfig
from .app_logger import AppLogger

class Downloader:
    """Отвечает за скачивание файла по URL."""
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, url: str, destination: Path):
        """
        Скачивает файл, если он еще не существует.
        :param url: URL для скачивания.
        :param destination: Путь для сохранения файла.
        """
        if destination.exists():
            self.log.info(f"Файл '{destination.name}' уже существует. Скачивание пропущено.")
            return

        self.log.info(f"Скачивание файла с {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.log.info(f"Файл '{destination.name}' успешно скачан.")
        except requests.exceptions.RequestException as e:
            self.log.error(f"Ошибка при скачивании файла: {e}")
            raise # Передаем ошибку выше

  