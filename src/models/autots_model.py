# src/models/autots_model.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from autots import AutoTS
from typing import Dict

from ..model_factory import BaseModel
from ..app_logger import AppLogger

class AutoTSModel(BaseModel):
    """
    Класс-обертка для библиотеки AutoTS.

    AutoTS автоматически находит лучший ансамбль моделей временных рядов.
    Примечание: Текущая реализация поддерживает только ОДНУ целевую переменную.
    Если в конфиге указано несколько таргетов, будет использован только первый.
    """
    def __init__(self, model_params: Dict):
        super().__init__(model_params)
        self.log = AppLogger()
        self.target_column_names = None # Будет определен при обучении 

        # Задаем параметры по умолчанию, которые можно переопределить в YAML
        default_params = {
            'frequency': 'D',
            'prediction_interval': 0.9,
            'ensemble': 'auto',
            'model_list': 'superfast',  # 'superfast', 'default', 'fast', 'all'
            'max_generations': 5,
            'num_validations': 2,
            'verbose': 1 # Отключаем собственный логгер AutoTS
        }
        # Обновляем дефолтные параметры теми, что пришли из конфига
        final_params = {**default_params, **self.params}

        # validation_metric - это параметр для .fit(), а не для __init__().
        # Удаляем его из словаря, чтобы избежать TypeError.
        if 'validation_metric' in final_params:
            del final_params['validation_metric']

        self.model = AutoTS(**final_params)

    def train(self, data_dict: Dict, train_params: Dict | None = None) -> Dict:
        """
        Обучает модель AutoTS.

        Args:
            data_dict (Dict): Словарь с данными. AutoTS требует 'X_train_df' и 'y_train_df'.
            train_params (Dict | None): Параметры обучения (игнорируются для AutoTS).

        Returns:
            Dict: Пустой словарь, так как AutoTS не возвращает историю.
        """
        X_train_df = data_dict.get('X_train_df')
        y_train_df = data_dict.get('y_train_df')

        if X_train_df is None or y_train_df is None:
            raise ValueError("Для AutoTSModel необходимы 'X_train_df' и 'y_train_df'.")

        # Определяем имя целевой колонки
        if not y_train_df.columns.empty:
            self.target_column_names = y_train_df.columns.tolist()

            ### self.target_column_name = y_train_df.columns[0]
            ### if len(y_train_df.columns) > 1:
            ###     self.log.warning(
            ###         f"AutoTS поддерживает только один таргет. "
            ###         f"Используется первый: '{self.target_column_name}'."
            ###     )
        else:
            raise ValueError("В 'y_train_df' отсутствуют целевые колонки.")

        ### ### ### # AutoTS требует DataFrame с датой в виде колонки, а не индекса
        ### ### ### full_train_df = pd.concat([X_train_df, y_train_df], axis=1)
        ### ### ### full_train_df.reset_index(inplace=True)
        ### ### 
        ### ### # Готовим основной DataFrame только с датой и таргетом
        ### ### train_df_for_fit = y_train_df.copy()
        ### ### train_df_for_fit.reset_index(inplace=True)
        ### ### 
        ### ### # Также сбрасываем индекс у регрессоров, чтобы они были консистентны
        ### ### X_train_df_reset = X_train_df.copy()
        ### ### X_train_df_reset.reset_index(drop=True, inplace=True)
        ### ### 
        ### ### # Переименовываем индекс в 'Date' для AutoTS
        ### ### date_col_name = train_df_for_fit.columns[0]
        ### 
        ### # Получаем имена регрессоров (все, кроме даты и таргета)
        ### # regressor_cols = [col for col in X_train_df.columns]
        ### 
        ### # AutoTS требует регулярный временной ряд. Выравниваем данные.
        ### y_train_df_resampled = y_train_df.resample('D').ffill()
        ### X_train_df_resampled = X_train_df.resample('D').ffill()
        ### 
        ### # AutoTS требует, чтобы дата была колонкой, а не индексом.
        ### ###### train_df_for_fit = y_train_df.reset_index()
        ### train_df_for_fit = y_train_df_resampled.reset_index()
        ### date_col_name = train_df_for_fit.columns[0]


        # Подготовка данных к "длинному" формату для AutoTS
        # 1. Объединяем таргеты и регрессоры, сохраняя DatetimeIndex
        full_train_df = pd.concat([y_train_df, X_train_df], axis=1)
        full_train_df.reset_index(inplace=True)
        date_col_name = full_train_df.columns[0]
        
        # 2. Преобразуем DataFrame из "широкого" в "длинный" формат
        # Все колонки, кроме даты и регрессоров, считаются идентификаторами
        id_vars = [date_col_name] + X_train_df.columns.tolist()
        
        df_long = full_train_df.melt(
            id_vars=id_vars,
            value_vars=self.target_column_names,
            var_name='series_id', # Новая колонка с именами таргетов
            value_name='target_value'   # Новая колонка со значениями таргетов
        )

        self.log.info(f"Запуск AutoTS.fit... Цели: {self.target_column_names}, регрессоры: {len(X_train_df.columns)} шт.")

        ### # Извлекаем validation_metric из исходных параметров
        ### validation_metric = self.params.get('validation_metric', 'smape')

        self.model.fit(
            df_long,
            # train_df_for_fit,
            date_col=date_col_name,
            value_col='target_value',
            # value_col=self.target_column_names,
            id_col='series_id',
            future_regressor=X_train_df
            # future_regressor=X_train_df_resampled
            ### validation_metric=validation_metric
        )
        
        self.log.info("Обучение AutoTS завершено.")
        self.log.info(f"Лучшая модель по версии AutoTS: {self.model.best_model_name}")
        self.log.info(str(self.model)) # Выводим детальную информацию о модели
        
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Делает предсказания на N шагов вперед.

        Args:
            X (pd.DataFrame): DataFrame с признаками для тестовой выборки.
                              Используется для определения длины прогноза и
                              как будущие регрессоры.

        Returns:
            np.ndarray: NumPy массив с предсказаниями.
        """
        if self.model is None or self.target_column_names is None:
            raise RuntimeError("Модель не обучена. Вызовите .train() перед .predict()")
        
        forecast_length = len(X)
        
        # Регрессоры для предсказания также должны быть выровнены по частоте
        X_resampled = X.resample('D').ffill()
        
        # Передаем будущие значения регрессоров
        predictions = self.model.predict(
            forecast_length=forecast_length,
            future_regressor=X_resampled
        )
        
        # Возвращаем только столбец с прогнозом
        return predictions.forecast.to_numpy()

    def save(self, path: Path) -> None:
        """Сохраняет обученную модель AutoTS."""
        if self.model:
            joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> 'AutoTSModel':
        """Загружает модель AutoTS."""
        model_instance = cls({}) # Создаем экземпляр с пустыми параметрами
        model_instance.model = joblib.load(path)
        # После загрузки нужно восстановить имя целевой колонки
        if hasattr(model_instance.model, 'df_wide_numeric') and not model_instance.model.df_wide_numeric.empty:
            model_instance.target_column_names = model_instance.model.df_wide_numeric.columns.tolist()
        return model_instance