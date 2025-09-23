# src/statistical_analyzer.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from .app_config import AppConfig
from .app_logger import AppLogger

class StatisticalAnalyzer:
    """
    Отвечает за статистический анализ и трансформацию временных рядов
    для приведения их к стационарному виду и обработки артефактов.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        """
        Инициализирует анализатор.

        Args:
            cfg (AppConfig): Глобальный конфигурационный объект.
            log (AppLogger): Экземпляр логгера.
        """
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Выполняет диагностику временного ряда (Фаза 1).
        Анализирует весь DataFrame на предмет стационарности, тренда и т.д.

        Args:
            df (pd.DataFrame): Исходный DataFrame с ценами.

        Returns:
            dict: Словарь с результатами анализа (global_insights).
        """
        self.log.info("Фаза 1: Запуск статистической диагностики...")
        
        close_series = df['Close']
        
        # 1. Тест на стационарность
        adf_result = adfuller(close_series.dropna())
        is_stationary = adf_result[1] < 0.05 # p-value < 0.05
        
        self.log.info(f"  ADF-тест на стационарность (p-value): {adf_result[1]:.4f}. Стационарный: {is_stationary}")
        
        # 2. Расчет показателя Хёрста
        hurst_exponent = self._calculate_hurst(close_series)
        if not np.isnan(hurst_exponent):
            if hurst_exponent > 0.5: hurst_interpretation = "Трендовый"
            elif hurst_exponent < 0.5: hurst_interpretation = "Возврат к среднему"
            else: hurst_interpretation = "Случайное блуждание"
            self.log.info(f"  Показатель Хёрста: {hurst_exponent:.4f} ({hurst_interpretation})")
        else:
            self.log.warning("  Не удалось рассчитать показатель Хёрста (слишком короткий ряд).")

        # 3. Тест на автокорреляцию (Льюнга-Бокса) на доходностях
        returns = close_series.pct_change().dropna()
        ljung_box_result = acorr_ljungbox(returns, lags=[21], return_df=True)
        lb_pvalue = ljung_box_result['lb_pvalue'].iloc[0]
        has_autocorrelation = lb_pvalue < 0.05
        self.log.info(f"  Тест Льюнга-Бокса на автокорреляцию (p-value): {lb_pvalue:.4f}. Автокорреляция: {has_autocorrelation}")

        global_insights = {
            "adf_pvalue": adf_result[1],
            "is_stationary": is_stationary,
            ### "has_trend": not is_stationary # Упрощенное допущение: если не стационарен, значит есть тренд
            "hurst_exponent": hurst_exponent,
            "lb_pvalue": lb_pvalue,
            "has_autocorrelation": has_autocorrelation
        }
        
        self.log.info("Статистическая диагностика завершена.")
        return global_insights

    def fit(self, train_df: pd.DataFrame, global_insights: dict) -> None:
        """
        Обучает параметры для трансформаций (Фаза 2, часть 1).
        Использует ТОЛЬКО тренировочные данные для расчета границ выбросов,
        параметров тренда и т.д.

        Args:
            train_df (pd.DataFrame): Тренировочный DataFrame.
            global_insights (dict): Результаты из фазы анализа.
        """
        self.log.info("Обучение параметров для статистических трансформаций на train-выборке...")
        # Пока заглушка, здесь будет расчет границ для выбросов.
        pass

    def transform(self, df: pd.DataFrame, global_insights: dict) -> pd.DataFrame:
        """
        Применяет обученные трансформации к данным (Фаза 2, часть 2).

        Args:
            df (pd.DataFrame): DataFrame для трансформации (train, val или test).
            global_insights (dict): Результаты из фазы анализа.

        Returns:
            pd.DataFrame: Трансформированный DataFrame.
        """
        self.log.info(f"Применение статистических трансформаций к выборке формы {df.shape}...")
        df_copy = df.copy()
        
        # 1. Логарифмирование OHLCV (применяется всегда)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_copy.columns:
                # Добавляем небольшую константу, чтобы избежать log(0) для Volume
                df_copy[col] = np.log(df_copy[col] + 1e-8)
        
        #### 2. Применение дифференцирования (если ряд не стационарен)
        ###if not global_insights.get("is_stationary", True):
        ###    self.log.info("  Применение первой разности (дифференцирование) для OHLC...")
        ###    for col in ['Open', 'High', 'Low', 'Close']:
        ###        if col in df_copy.columns:
        ###            df_copy[col] = df_copy[col].diff()
        
        # Очистка NaN, появившихся после .diff()
        df_copy.dropna(inplace=True)
        
        return df_copy
    
    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Рассчитывает показатель Хёрста для временного ряда.

        Args:
            series (pd.Series): Временной ряд (например, цены закрытия).

        Returns:
            float: Значение показателя Хёрста.
        """
        if len(series) < 100:
            return np.nan # Недостаточно данных для надежного расчета

        lags = range(2, 100)
        # Рассчитываем стандартизированный размах
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        # Используем полифит для нахождения наклона в лог-лог масштабе
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return poly[0] * 2.0

  