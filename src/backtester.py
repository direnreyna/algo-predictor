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

    def run(self, experiment_cfg: ExperimentConfig, predictions: np.ndarray, test_df: pd.DataFrame, scaler: object) -> dict:
        """
        Запускает бэктест.
        :param experiment_cfg: Конфигурация текущего эксперимента.
        :param predictions: Массив с предсказаниями модели (N, 2).
        :param test_df: Полный масштабированный тестовый датафрейм.
        :param scaler: Обученный скейлер для обратного преобразования.
        :return: Словарь с финансовыми метриками (Sharpe, Drawdown и т.д.).
        """
        self.log.info(f"Запуск финансового бэктеста для задачи типа '{experiment_cfg.task_type}'...")
        
        # 1. Восстанавливаем исходные, не масштабированные цены ##ДОБАВЛЕН БЛОК
        # Создаем копию, чтобы избежать SettingWithCopyWarning
        unscaled_test_df = test_df.copy()
        
        # Определяем колонки, которые были масштабированы (все кроме целевых)
        feature_cols = [col for col in unscaled_test_df.columns if not col.startswith('target_')]
        
        # Применяем обратное преобразование
        unscaled_test_df[feature_cols] = scaler.inverse_transform(unscaled_test_df[feature_cols])
        
        # Извлекаем цены закрытия. Они нужны для vbt.Portfolio.
        # test_df индексирован по дате, поэтому close_prices будет правильной временной серией.
        close_prices = unscaled_test_df['Close']

        # Для моделей, использующих окна, количество предсказаний меньше, чем исходный test set.
        # Необходимо синхронизировать close_prices с предсказаниями.
        if len(predictions) < len(close_prices):
            offset = len(close_prices) - len(predictions)
            close_prices = close_prices.iloc[offset:]

        if experiment_cfg.task_type == "classification":
            ### # Сигнал 2=UP (Buy), 1=SIDEWAYS (Hold), 0=DOWN (Sell)
            ### predictions = pd.Series(np.random.randint(0, 3, size=num_days), index=close_prices.index)

            # Преобразуем numpy-предсказания в именованный pd.Series для удобства
            predictions_series = pd.Series(predictions, index=close_prices.index)

            # 2. Преобразование предсказаний в торговые сигналы
            entries = predictions_series == 2
            exits = predictions_series == 0

        elif experiment_cfg.task_type == "regression":
            ### # Симулируем предсказания high и low для следующего дня
            ### predicted_high = close_prices * (1 + np.random.uniform(0.005, 0.02, size=num_days))
            ### predicted_low = close_prices * (1 - np.random.uniform(0.005, 0.02, size=num_days))

            # Извлекаем реальные предсказания модели
            predicted_high = pd.Series(predictions[:, 0], index=close_prices.index)
            predicted_low = pd.Series(predictions[:, 1], index=close_prices.index)

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
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": -1.0,
                "total_return_pct": 0.0
            }
        
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