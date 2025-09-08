# src.data_saver.py

import numpy as np
import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger

class DataSaver:
    """
    Отвечает за сохранение и загрузку обработанных данных (кеширование).
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataSaver, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, cfg: AppConfig, log: AppLogger):
        if self._initialized:
            return
        self.cfg = cfg
        self.log = log
        self.save_path = self.cfg.PREPARED_DATA_PATH
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")
        self._initialized = True

    def save(self, **kwargs: pd.DataFrame):
        """
        Сохраняет переданные DataFrame в один .npz файл.

        :param kwargs: Именованные аргументы, где ключ - имя выборки
                       (напр., 'train'), а значение - DataFrame.
        """
        self.log.info(f"Сохранение обработанных данных в '{self.save_path.name}'...")
        try:
            np.savez_compressed(self.save_path, **{name: df.to_numpy() for name, df in kwargs.items()})
            self.log.info("Данные успешно сохранены.")
        except Exception as e:
            self.log.error(f"Ошибка при сохранении данных: {e}")
            raise

    def load(self) -> dict[str, np.ndarray]:
        """
        Загружает данные из .npz файла.

        :return: Словарь, где ключ - имя выборки, значение - NumPy массив.
        """
        self.log.info(f"Загрузка обработанных данных из '{self.save_path.name}'...")
        try:
            data = np.load(self.save_path, allow_pickle=True)
            self.log.info("Данные успешно загружены из кеша.")
            return {key: data[key] for key in data.files}
        except Exception as e:
            self.log.error(f"Ошибка при загрузке данных: {e}")
            raise
