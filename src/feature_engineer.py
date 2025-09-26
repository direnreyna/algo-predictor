# src.feature_engineer.py

import numpy as np
import pandas as pd
import pandas_ta as ta
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

    def run(self, df: pd.DataFrame, experiment_cfg: ExperimentConfig) -> pd.DataFrame:
        """
        Динамически генерирует набор технических индикаторов.
        Разделяет индикаторы на стандартные (pandas-ta) и кастомные (методы класса).
        
        :param df: Исходный DataFrame.
        :param experiment_cfg: Список технических индикаторов.
        :return: DataFrame, обогащенный новыми признаками.
        """
        feature_set_name = experiment_cfg.common_params.get("feature_set_name")
        self.log.info(f"Генерация набора признаков '{feature_set_name}'...")

        # 1. Получаем список индикаторов из 
        feature_list = self.cfg.FEATURE_SETS.get(feature_set_name)
        if not feature_list:
            raise ValueError(f"Набор признаков '{feature_set_name}' не найден в AppConfig.")

        df_copy = df.copy()

        # Блок обработки DATETIME признаков
        datetime_feature_names = self.cfg.FEATURE_SETS.get("datetime_features", [])
        requested_dt_features = [f for f in feature_list if f in datetime_feature_names]
        if requested_dt_features:
            self.log.info(f"Расчет {len(requested_dt_features)} временных признаков...")
            self._calculate_datetime_features(df_copy)

        # 2. Разделяем на стандартные и кастомные
        standard_indicators = []
        custom_indicators = []
        # Исключаем 'Open', 'High', 'Low', 'Close', 'Volume' из обработки
        ohlcv = self.cfg.FEATURE_SETS.get('ohlcv', []) 
        
        for name in feature_list:
            if name in ohlcv: continue # Пропускаем базовые OHLCV колонки
            if name in datetime_feature_names: continue

            # Проверяем. существует ли аттрибут (метод) с указанным именем в текущем классе (это кастомный метод)
            if hasattr(self, f"_calculate_{name}"):
                custom_indicators.append(name)
            # ... если не существует, значит это должен быть стандартный индикатор для pandas-ta.
            else:
                standard_indicators.append(name)
        
        # 3. Генерируем стандартные индикаторы
            if standard_indicators:
                self.log.info(f"Расчет {len(standard_indicators)} стандартных индикаторов...")
                
                for indicator_str in standard_indicators:
                    try:
                        ### # --- Диагностический блок 1 ---
                        ### before_nan_count = df_copy.isna().sum().sum()
                        ### # --- Конец блока ---

                        # --- Устойчивый блок с проверкой ---
                        cols_before = set(df_copy.columns)
                        self._apply_indicator(df_copy, indicator_str)
                        cols_after = set(df_copy.columns)
                        new_cols = list(cols_after - cols_before)

                        # Проверяем новые колонки на полную пустоту
                        sanitized_new_cols = new_cols.copy()          # Список новых колонок для проверки на NaN
                        for col in new_cols:
                            # Если среди новых колонок есть колонка целиком состоящая из NaN
                            if df_copy[col].isna().all():
                                self.log.warning(f"Индикатор '{indicator_str}' создал полностью пустую колонку '{col}'. Колонка будет удалена.")
                                # Удаляем такую колонку
                                df_copy.drop(columns=[col], inplace=True)
                                # Удаляем такую колонку из списка новых колонок для проверки на NaN
                                sanitized_new_cols.remove(col)
                        
                        # Проверяем на "вредность" (более 50% NaN в совокупности новых неудаленных колонок)
                        if sanitized_new_cols: # Если после чистки "на 100% NaN" что-то осталось
                            temp_df = df_copy[sanitized_new_cols]
                            # Количество строк, содержащих NaN
                            n_damaged_rows = temp_df.isna().any(axis=1).sum()
                            # Процент строк, содержащих NaN
                            damage_ratio = n_damaged_rows / len(df_copy)
                            
                            if damage_ratio > 0.5:
                                self.log.warning(
                                    f"Индикатор '{indicator_str}' признан 'вредным' "
                                    f"(портит {damage_ratio:.1%} строк). Колонки {sanitized_new_cols} будут удалены."
                                )
                                # Удаляем все такие колонки
                                df_copy.drop(columns=sanitized_new_cols, inplace=True)
                        # --- Конец блока ---

                        ### # --- Диагностический блок 2 ---
                        ### after_nan_count = df_copy.isna().sum().sum()
                        ### new_nans = after_nan_count - before_nan_count
                        ### self.log.info(f"  -> {indicator_str}: добавлено {new_nans} NaN.")
                        ### # --- Конец блока ---

                    except Exception as e:
                        self.log.error(f"Ошибка при расчете индикатора '{indicator_str}': {e}")
                        # Можно либо проигнорировать, либо прервать выполнение
                        raise

        # 4. Генерируем кастомные индикаторы
        if custom_indicators:
            self.log.info(f"Расчет {len(custom_indicators)} кастомных индикаторов...")
            for name in custom_indicators:
                method_to_call = getattr(self, f"_calculate_{name}")
                df_copy = method_to_call(df_copy)

        ### # --- Диагностика: Состояние NaN перед финальной очисткой ---
        ### self.log.info(f"Состояние NaN перед финальной очисткой (Total={df_copy.isna().sum().sum()}):\n{df_copy.isna().sum()}")
        ### # --- Конец Диагностики ---

        # 5. Генерация лаговых признаков (если указано в конфиге)
        lag_config = experiment_cfg.common_params.get("lag_features")
        if lag_config:
            self.log.info("Генерация лаговых признаков...")
            self._calculate_lag_features(df_copy, lag_config)

        # 6. Финальная очистка: заполнение и отсечение периода прогрева
        # 6.1 Заполняем пропуски в середине данных (от выходных и т.д.)
        initial_nan_count = df_copy.isna().sum().sum()
        if initial_nan_count > 0:
            df_copy.ffill(inplace=True)
            filled_nan_count = initial_nan_count - df_copy.isna().sum().sum()
            self.log.info(f"Заполнение NaN (forward fill): заполнено {filled_nan_count} значений.")

        # 6.2 Отсекаем начальный период "прогрева", где ffill не смог помочь
        # Считаем, сколько NaN осталось в начале каждой колонки. Логика:
        # ===============================================================
        # df_copy.notna() = делает все NaN = False, остальные = True.
        # cumsum() = кумулятивная сумма. Все первые False дадут 0, первый True изменил сумму на > 0.
        # == 0 создаст маску, где каждый 0 станет True (нули только в начале)
        # .sum() - сосчитает количество начальных 0.
        # ===============================================================
        initial_nans_per_col = (df_copy.notna().cumsum() == 0).sum()
        max_lookback = initial_nans_per_col.max()
        
        initial_rows = len(df_copy)
        if max_lookback > 0:
            self.log.info(f"Максимальный период прогрева: {max_lookback} баров. Отсекаем...")
            df_copy = df_copy.iloc[max_lookback:]
        
        final_rows = len(df_copy)
        self.log.info(f"Отсечение периода прогрева: удалено {initial_rows - final_rows} строк.")
        
        self.log.info(f"Генерация признаков завершена. Итоговая форма данных: {df_copy.shape}")
        return df_copy

    def _calculate_lag_features(self, df: pd.DataFrame, lag_config: dict) -> None:
        """
        Создает лаговые признаки для указанных колонок.
        Модифицирует DataFrame на месте (inplace).

        Args:
            df (pd.DataFrame): DataFrame для добавления признаков.
            lag_config (dict): Словарь, где ключ - имя колонки,
                               а значение - список лагов.
        """
        for col_name, periods in lag_config.items():
            if col_name not in df.columns:
                self.log.warning(f"Колонка '{col_name}' для создания лагов не найдена. Пропускается.")
                continue
            
            for p in periods:
                new_col_name = f"{col_name}_lag_{p}"
                df[new_col_name] = df[col_name].shift(p)

    def _apply_indicator(self, df: pd.DataFrame, indicator_str: str) -> None:
        """
        Применяет один индикатор к DataFrame.
        Парсит строку 'name_param1_param2' и вызывает df.ta.name(param1, param2).
        """
        parts = indicator_str.split('_')
        name = parts[0]
        # Преобразуем строковые параметры в числа (int или float)
        params = [float(p) if '.' in p else int(p) for p in parts[1:]]
        
        indicator_func = getattr(df.ta, name)
        # Передаем параметры позиционно
        indicator_func(*params, append=True)

    def _calculate_custom_test_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Пример кастомного индикатора]
        Рассчитывает среднее значение свечи (OHLC/4).
        Этот метод служит шаблоном для добавления новых, сложных индикаторов,
        которых нет в библиотеке pandas-ta.

        :param df: DataFrame с обязательными колонками 'Open', 'High', 'Low', 'Close'.
        :return: DataFrame с добавленной колонкой 'custom_test_indicator'.
        """
        df['custom_test_indicator'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        return df
    
    def _calculate_datetime_features(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает и добавляет в DataFrame циклические признаки времени и год.
        Модифицирует DataFrame на месте (inplace).

        Args:
            df (pd.DataFrame): DataFrame с DatetimeIndex.
        """

        # Убедимся, что индекс имеет тип DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            self.log.warning("DataFrame не имеет DatetimeIndex. Признаки времени не будут рассчитаны.")
            return
        
        # День года (с учетом високосных лет)
        day_of_year = df.index.dayofyear
        days_in_year = np.where(df.index.is_leap_year, 366, 365)
        df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
        df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)

        # День недели
        day_of_week = df.index.dayofweek # Понедельник=0, Воскресенье=6
        df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Месяц
        month = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Год (как линейный тренд)
        df['year'] = df.index.year
        
        # День месяца (с учетом реального количества дней в месяце)
        day = df.index.day
        days_in_month = df.index.days_in_month
        df['day_sin'] = np.sin(2 * np.pi * (day - 1) / days_in_month)
        df['day_cos'] = np.cos(2 * np.pi * (day - 1) / days_in_month)

        # Квартал года
        quarter = df.index.quarter
        df['quarter_sin'] = np.sin(2 * np.pi * (quarter - 1) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * (quarter - 1) / 4)
