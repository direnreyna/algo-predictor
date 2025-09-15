# src.feature_engineer.py

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
        self.log.info(f"Генерация набора признаков '{experiment_cfg.feature_set_name}'...")

        
        # 1. Получаем список индикаторов из конфига
        feature_list = self.cfg.FEATURE_SETS.get(experiment_cfg.feature_set_name)
        if not feature_list:
            raise ValueError(f"Набор признаков '{experiment_cfg.feature_set_name}' не найден в AppConfig.")

        df_copy = df.copy()

        # 2. Разделяем на стандартные и кастомные
        standard_indicators = []
        custom_indicators = []
        # Исключаем 'Open', 'High', 'Low', 'Close', 'Volume' из обработки
        ohlcv = self.cfg.FEATURE_SETS.get('ohlcv', []) 
        
        for name in feature_list:
            if name in ohlcv: continue # Пропускаем базовые OHLCV колонки

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

        # 5. Финальная очистка: заполнение и отсечение периода прогрева ##ДОБАВЛЕН БЛОК
        # 5.1 Заполняем пропуски в середине данных (от выходных и т.д.)
        initial_nan_count = df_copy.isna().sum().sum()
        if initial_nan_count > 0:
            df_copy.ffill(inplace=True)
            filled_nan_count = initial_nan_count - df_copy.isna().sum().sum()
            self.log.info(f"Заполнение NaN (forward fill): заполнено {filled_nan_count} значений.")

        # 5.2 Отсекаем начальный период "прогрева", где ffill не смог помочь
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