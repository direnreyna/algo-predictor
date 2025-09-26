# src/data_enricher.py

import pandas as pd
from typing import List

from .app_config import AppConfig
from .app_logger import AppLogger
from .file_loader import FileLoader

class DataEnricher:
    """
    Отвечает за обогащение основного временного ряда внешними данными.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger, file_loader: FileLoader):
        self.cfg = cfg
        self.log = log
        self.file_loader = file_loader
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, main_df: pd.DataFrame, enrichment_config: List[dict]) -> pd.DataFrame:
        """
        Последовательно присоединяет к основному DataFrame внешние источники данных.

        Args:
            main_df (pd.DataFrame): Основной DataFrame с ценами.
            enrichment_config (List[dict]): Список словарей из YAML-конфига,
                                             описывающих источники для обогащения.

        Returns:
            pd.DataFrame: Обогащенный DataFrame.
        """
        if not enrichment_config:
            return main_df

        self.log.info(f"Начало обогащения данных. {len(enrichment_config)} источников.")
        enriched_df = main_df.copy()

        for source in enrichment_config:
            filename = source['filename']
            data_type = source['data_type']
            
            # 1. Загружаем внешний DataFrame
            external_df = self.file_loader.read_csv(filename, data_type=data_type)

            # 2. Объединяем по индексу (дате)
            enriched_df = pd.merge(enriched_df, external_df, how='left', left_index=True, right_index=True)

            # 3. Заполняем пропуски на основе типа данных
            if data_type in ['interest_rate', 'cpi']:
                enriched_df[external_df.columns] = enriched_df[external_df.columns].ffill()
        
        self.log.info("Обогащение данных завершено.")
        return enriched_df

  