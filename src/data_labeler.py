# src.data_labeler.py

import pandas as pd
import numpy as np
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class DataLabeler:
    """
    Отвечает за создание целевой переменной (меток классов).
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, df:pd.DataFrame, experiment_cfg: ExperimentConfig) -> pd.DataFrame:
        """
        Создает столбец 'target' с классами (0, 1, 2)
        на основе параметров HORIZON и THRESHOLD из конфига.

        :param df: DataFrame с признаками.
        :return: DataFrame с добавленным столбцом 'target'.
        """

        self.log.info(f"Создание целевой переменной для задачи типа '{self.cfg.TASK_TYPE}'...")
        if self.cfg.TASK_TYPE == "classification":

            df_copy = df.copy()
            horizon = experiment_cfg.labeling_horizon
            threshold = self.cfg.THRESHOLD

            # 1. Расчет будущих изменений цены для каждой точки в пределах горизонта
            future_returns = pd.concat([
                (df_copy['Close'].shift(-i) / df_copy['Close'] - 1) for i in range(1, horizon + 1)
            ], axis=1)
            future_returns.columns = list(range(1, horizon + 1))

            # 2. Определение моментов касания барьеров
            upper_barrier_hits = (future_returns > threshold).idxmax(axis=1)
            lower_barrier_hits = (future_returns < -threshold).idxmax(axis=1)
            
            # 3. Маски для случаев, когда барьеры не были достигнуты (idxmax вернет 0, т.е. первый столбец)
            no_upper_hit_mask = (future_returns.loc[upper_barrier_hits.index, 1] <= threshold)
            upper_barrier_hits[no_upper_hit_mask] = np.nan

            no_lower_hit_mask = (future_returns.loc[lower_barrier_hits.index, 1] >= -threshold)
            lower_barrier_hits[no_lower_hit_mask] = np.nan

            # 4. Определение итогового события
            # Заполняем NaN большим числом, чтобы min() работал корректно
            first_hit_time = pd.concat([upper_barrier_hits, lower_barrier_hits], axis=1).min(axis=1)
            
            # 5. Присвоение финальных меток
            # Инициализируем 'target' значением для бокового движения
            df_copy['target'] = 1
            
            # Верхний барьер достигнут первым
            df_copy.loc[first_hit_time == upper_barrier_hits, 'target'] = 2 # UP
            
            # Нижний барьер достигнут первым
            df_copy.loc[first_hit_time == lower_barrier_hits, 'target'] = 0 # DOWN
            
            # Вертикальный барьер (ни один из горизонтальных не достигнут)
            # В этом случае метка уже 1 (SIDEWAYS), дополнительная логика не требуется.
            # Можно добавить логику для определения направления на последнем шаге,
            # но классический TBM оставляет это как нейтральный исход.

            df = df_copy
            df.dropna(inplace=True)

        elif self.cfg.TASK_TYPE == "regression":
            # Другая логика, например, предсказываем будущее изменение цены
            df['target'] = df['Close'].shift(-experiment_cfg.labeling_horizon) / df['Close'] - 1
            df.dropna(inplace=True)
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.cfg.TASK_TYPE}")
        
        return df