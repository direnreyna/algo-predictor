# src.data_preparer.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger

class DataPreparer:
    """
    Отвечает за всю предобработку данных:
    - Расчет технических индикаторов.
    - Создание целевой переменной (классов).
    - Очистку/нормализацию.
    - Разделение на выборки.
    - сохранение в .npz.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет полный цикл предобработки данных.

        :param df: Исходный DataFrame для обработки.
        :return: DataFrame с добавленными признаками и целевой переменной.
        """
        self.log.info("Предобработка данных...")
        return df