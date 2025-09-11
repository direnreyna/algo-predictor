# src.feature_engineer.py

import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class FeatureEngineer:
    """
    Отвечает за расчет и добавление технических индикаторов в DataFrame.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df:pd.DataFrame, experiment_cfg: ExperimentConfig) -> pd.DataFrame:
        """
        Добавляет в DataFrame все технические индикаторы.

        :param df: Исходный DataFrame.
        :return: DataFrame, обогащенный новыми признаками.
        """
        self.log.info("Добавление технических индикаторов в DataFrame...")
        return df