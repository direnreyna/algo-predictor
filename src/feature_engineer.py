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
                        self._apply_indicator(df_copy, indicator_str)
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

        # 5. Очистка от NaN
        initial_rows = len(df_copy)
        df_copy.dropna(inplace=True)
        final_rows = len(df_copy)
        self.log.info(f"Очистка от NaN: удалено {initial_rows - final_rows} строк.")
        
        self.log.info(f"Генерация признаков завершена. Итоговая форма данных: {df_copy.shape}")
        return df_copy
    
###     def _parse_indicator_string(self, indicator_str: str) -> tuple[str, dict]:
###         """
###         Парсит строку индикатора в имя и словарь параметров.
###         Пример: 'RSI_14' -> ('rsi', {'length': 14})
###                  'MACD_12_26_9' -> ('macd', {'fast': 12, 'slow': 26, 'signal': 9})
###         """
###         parts = indicator_str.split('_')
###         name = parts[0]
###         params = parts[1:]
###         
###         # pandas-ta использует стандартные имена для параметров
###         # Это маппинг для самых распространенных из них
###         param_names_map = {
###             'RSI': ['length'],
###             'MACD': ['fast', 'slow', 'signal'],
###             'STOCHk': ['k', 'd', 'smooth_k'],
###             'BBP': ['length', 'std'],
###             'ATR': ['length'],
###             'OBV': [],
###         }
###         
###         if name.upper() not in param_names_map:
###             # Для неизвестных индикаторов предполагаем универсальный параметр 'length'
###             if len(params) == 1:
###                  return name, {'length': int(params[0])}
###             raise ValueError(f"Неизвестная структура параметров для индикатора: {name}")
### 
###         param_names = param_names_map[name.upper()]
###         if len(param_names) != len(params):
###             raise ValueError(f"Неверное количество параметров для {name}. Ожидается {len(param_names)}, получено {len(params)}")
### 
###         # Преобразуем строковые параметры в числа (int или float)
###         typed_params = []
###         for p in params:
###             typed_params.append(float(p) if '.' in p else int(p))
### 
###         return name, dict(zip(param_names, typed_params))

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