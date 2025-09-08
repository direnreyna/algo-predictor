# src.data_labeler.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger

class DataLabeler:
    """
    Отвечает за создание целевой переменной (меток классов).
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Создает столбец 'target' с классами (0, 1, 2)
        на основе параметров HORIZON и THRESHOLD из конфига.

        :param df: DataFrame с признаками.
        :return: DataFrame с добавленным столбцом 'target'.
        """

        self.log.info(f"Создание целевой переменной для задачи типа '{self.cfg.TASK_TYPE}'...")
        if self.cfg.TASK_TYPE == "classification":
            # Ваша логика с HORIZON, THRESHOLD и созданием 'target'
            pass
        elif self.cfg.TASK_TYPE == "regression":
            # Другая логика, например, предсказываем будущее изменение цены
            df['target'] = df['Close'].shift(-self.cfg.HORIZON) / df['Close'] - 1
            df.dropna(inplace=True)
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.cfg.TASK_TYPE}")
        
        return df