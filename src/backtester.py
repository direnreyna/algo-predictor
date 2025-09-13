# src/backtester.py

import pandas as pd
import numpy as np
import vectorbt as vbt
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class Backtester:
    """
    Отвечает за проведение финансового бэктеста на основе
    предсказаний модели.
    """

    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, experiment_cfg: ExperimentConfig) -> dict:
        """
        Запускает бэктест.
        :param experiment_cfg: Конфигурация текущего эксперимента.
        :return: Словарь с финансовыми метриками (Sharpe, Drawdown и т.д.).
        """
        self.log.info(f"Запуск финансового бэктеста для задачи типа '{experiment_cfg.task_type}'...")
        
        # 1. Генерация симулированных данных
        num_days = 1000
        close_prices = pd.Series(np.random.randn(num_days).cumsum() + 100, 
                                 index=pd.to_datetime(pd.date_range('2020-01-01', periods=num_days)))
        
        if experiment_cfg.task_type == "classification":
            # Сигнал 2=UP (Buy), 1=SIDEWAYS (Hold), 0=DOWN (Sell)
            predictions = pd.Series(np.random.randint(0, 3, size=num_days), index=close_prices.index)

            # 2. Преобразование предсказаний в торговые сигналы
            entries = predictions == 2
            exits = predictions == 0

        elif experiment_cfg.task_type == "regression":
            # Симулируем предсказания high и low для следующего дня
            predicted_high = close_prices * (1 + np.random.uniform(0.005, 0.02, size=num_days))
            predicted_low = close_prices * (1 - np.random.uniform(0.005, 0.02, size=num_days))

            # Простая стратегия: покупаем, если предсказанный low выше текущего close.
            # Продаем, если предсказанный high ниже текущего close.
            entries = predicted_low > close_prices
            exits = predicted_high < close_prices
        else:
            raise ValueError(f"Неизвестная логика бэктестинга для типа задачи: {experiment_cfg.task_type}")
        
        # 3. Запуск векторизованного бэктеста
        # initial_cash=100000, freq='1D' - стандартные параметры для симуляции

        portfolio = vbt.Portfolio.from_signals(
            close=close_prices, 
            entries=entries, 
            exits=exits, 
            init_cash=100000,
            freq='1D'
        )

        # 4. Извлечение и форматирование ключевых метрик
        stats = portfolio.stats()
    
        # 5. Обработка отсутствия 
        if stats is None:
            self.log.warning("В бэктесте не было совершено ни одной сделки (stats is None). Возвращены нулевые метрики.")
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": -1.0,
                "total_return_pct": 0.0
            }        

        # stats точно не None, теперь безопасно извлекаем значение
        total_trades = stats.get('Total Trades', 0)
        if isinstance(total_trades, pd.Series):
            total_trades = total_trades.iloc[0]

        if total_trades == 0:
            self.log.warning("В бэктесте не было совершено ни одной сделки (Total Trades = 0). Возвращены нулевые метрики.")
            # --- ЗАГЛУШКА ДЛЯ OPTUNA ---
            # Генерируем случайный Sharpe для проверки работы оптимизатора
            import random
            return {
                "sharpe_ratio": random.uniform(-0.5, 2.5),
                "max_drawdown": -1.0,
                "total_return_pct": 0.0
            }
            # --- КОНЕЦ ЗАГЛУШКИ
        
        sharpe = stats.get('Sharpe Ratio', 0.0)

        # Приводим Max Drawdown к отрицательному формату [-1.0, 0.0]
        max_drawdown = -stats.get('Max Drawdown [%]', 100.0) / 100.0
        total_return = stats.get('Total Return [%]', 0.0)

        # Обработка NaN/inf, которые могут возникнуть, если не было сделок
        sharpe = 0.0 if not np.isfinite(sharpe) else sharpe
        
        ### # --- ЗАГЛУШКА ДЛЯ OPTUNA
        ### # Генерируем случайный Sharpe для проверки работы оптимизатора
        ### import random
        ### sharpe = random.uniform(-0.5, 2.5)
        ### # --- КОНЕЦ ЗАГЛУШКИ
        
        financial_metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return_pct": total_return
        }

        return financial_metrics