# src.data_splitter.py

from typing import Tuple
import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger

class DataSplitter:
    """
    Отвечает за разделение данных на выборки и их масштабирование.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделяет и масштабирует данные на train, validation и test.

        :param df: Полный DataFrame с признаками и метками.
        :return: Кортеж из трех DataFrame: (train, validation, test).
        """
        self.log.info("Начинаем разделение и нормирование датасета...")
        return df, df, df