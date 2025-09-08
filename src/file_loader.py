# src.file_loader.py

import os
import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger
from .downloader import Downloader
from .unpacker import Unpacker
from pathlib import Path

class FileLoader:
    """
    Оркестратор загрузки данных.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.downloader = Downloader(cfg, log)
        self.unpacker = Unpacker(cfg, log)
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self) -> pd.DataFrame:
        """
        Выполняет полный пайплайн:
        1. Проверяет наличие ЦЕЛЕВОГО файла.
        2. Если его нет, скачивает все нужные архивы/файлы.
        3. Распаковывает все архивы в папке data/ и удаляет их.
        4. Загружает ЦЕЛЕВОЙ файл в DataFrame.

        :return: DataFrame с исходными данными.
        """
        self.log.info("--- Начало процесса загрузки данных ---")
        
        #target_file_name = self.cfg.DATA_FILE
        target_file_path = self.cfg.DATA_DIR / self.cfg.DATA_FILE

        # ГЛАВНАЯ ПРОВЕРКА: Если целевой файл уже есть, ничего не делаем
        if target_file_path.exists():
            self.log.info(f"Целевой файл '{self.cfg.DATA_FILE}' уже существует.")
        else:
            self.log.info(f"Целевой файл '{self.cfg.DATA_FILE}' не найден. Начинаем подготовку.")
        
            # 1: Скачиваем все файлы, для которых есть URL
            self._download_all_sources()
        
            # 2: Ищем и распаковываем все архивы в папке data/
            self._unpack_all_archives()

        # 3: Читаем ЦЕЛЕВОЙ файл в DataFrame
        df = self._read_data(target_file_path)

        self.log.info("--- Процесс загрузки и подготовки файлов завершен ---")
        return df

    def _download_all_sources(self):
        """Скачивает все файлы, указанные в DOWNLOAD_URLS."""
        self.log.info("Проверка и скачивание исходных файлов...")
        for filename, url in self.cfg.DOWNLOAD_URLS.items():
            if url:
                filepath = self.cfg.DATA_DIR / filename
                self.downloader.run(url, filepath)

    def _unpack_all_archives(self):
        """Ищет все архивы в папке data/, распаковывает их и удаляет исходные архивы."""
        self.log.info("Поиск, распаковка и очистка архивов...")
        # Создаем копию списка, так как будем изменять содержимое папки в цикле
        for filename in list(os.listdir(self.cfg.DATA_DIR)):
            filepath = self.cfg.DATA_DIR / filename
            if filepath.is_file():
                unpacked = self.unpacker.run(filepath)
                if unpacked:
                    # Если распаковка была успешной, удаляем исходный архив
                    try:
                        os.remove(filepath)
                        self.log.info(f"Исходный архив '{filename}' удален.")
                    except OSError as e:
                        self.log.error(f"Не удалось удалить архив '{filename}': {e}")

    def _read_data(self, filepath: Path) -> pd.DataFrame:
        """Приватный метод для чтения CSV."""
        self.log.info(f"Чтение данных из файла '{filepath.name}'...")
        try:
            # Здесь можно добавить логику для разных форматов (xlsx, json...)
            df = pd.read_csv(filepath)
            self.log.info(f"Данные успешно загружены. Размер: {df.shape}")
            return df
        except FileNotFoundError:
            self.log.error(f"Файл не найден: {filepath}")
            raise
        except Exception as e:
            self.log.error(f"Ошибка при чтении файла: {e}")
            raise