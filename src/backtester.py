# src/backtester.py

import pandas as pd
import numpy as np
import vectorbt as vbt
import mlflow
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig
from .visualization_utils import VisualizationUtils
from .inverse_transformer import InverseTransformer

class Backtester:
    """
    Отвечает за проведение финансового бэктеста на основе
    предсказаний модели.
    """

    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, experiment_cfg: ExperimentConfig, predictions: np.ndarray, test_df: pd.DataFrame, inverse_transformer: InverseTransformer) -> dict:
        """
        Запускает бэктест.

        Args:
            experiment_cfg (ExperimentConfig): Конфигурация текущего эксперимента.
            predictions (np.ndarray): Массив с предсказаниями модели (масштабированными).
            test_df (pd.DataFrame): Полный МАСШТАБИРОВАННЫЙ тестовый датафрейм.
            inverse_transformer (InverseTransformer): Экземпляр для выполнения
                обратного преобразования предсказаний.

        Returns:
            Словарь с финансовыми метриками (Sharpe, Drawdown и т.д.).
        """
        task_type = experiment_cfg.common_params.get("task_type")
        self.log.info(f"Запуск финансового бэктеста для задачи типа '{task_type}'...")
             
        #### # Блок проверки наличия scaler
        #### if scaler:                            
        ####     # 1. Получаем немасштабированные предсказания
        ####     # Создаем временный массив нужной формы для inverse_transform
        ####     temp_array = np.zeros((len(predictions), len(test_df.columns)))
        #### 
        ####     # Находим индексы целевых колонок в test_df (на которых обучался scaler)
        ####     target_indices = [test_df.columns.get_loc(f"target_{name}") for name in experiment_cfg.common_params.get("targets")]
        ####     
        ####     # Вставляем предсказания в нужные места
        ####     for i, idx in enumerate(target_indices):
        ####         temp_array[:, idx] = predictions[:, i]
        #### 
        ####     # Делаем inverse_transform
        ####     unscaled_array = scaler.inverse_transform(temp_array)
        ####     
        ####     # Извлекаем только наши немасштабированные предсказания
        ####     unscaled_predictions = unscaled_array[:, target_indices]
        #### else:
        ####     # Если скейлера не было, предсказания уже в нужном масштабе
        ####     unscaled_predictions = predictions
        #### 
        #### ### # 1. Восстанавливаем исходные, не масштабированные цены
        #### ### # Создаем копию, чтобы избежать SettingWithCopyWarning
        #### ### unscaled_test_df = test_df.copy()
        #### ### 
        #### ### # Определяем ВСЕ числовые колонки, которые были масштабированы
        #### ### cols_to_unscale = unscaled_test_df.select_dtypes(include=np.number).columns.tolist()
        #### ### 
        #### ### # Применяем обратное преобразование
        #### ### unscaled_test_df[cols_to_unscale] = scaler.inverse_transform(unscaled_test_df[cols_to_unscale])
        #### 
        #### # Извлекаем цены закрытия. Они нужны для vbt.Portfolio.
        #### # test_df индексирован по дате, поэтому close_prices будет правильной временной серией.
        #### ### close_prices = unscaled_test_df['Close']
        #### close_prices = original_test_df['Close']

        # Шаг 1: Получаем предсказания в абсолютных значениях
        y_pred_abs = inverse_transformer.transform(predictions)

        # Шаг 2: Получаем оригинальные данные для бэктеста
        original_test_df = inverse_transformer.original_test_df
        close_prices = original_test_df['Close']

        # Для моделей, использующих окна, количество предсказаний меньше, чем исходный test set.
        # Необходимо синхронизировать close_prices с предсказаниями.
        offset = 0
        if len(predictions) < len(close_prices):
            offset = len(close_prices) - len(predictions)
            close_prices = close_prices.iloc[offset:]

        if task_type == "classification":
            # Преобразуем numpy-предсказания в именованный pd.Series для удобства
            predictions_series = pd.Series(predictions, index=close_prices.index)

            # 2. Преобразование предсказаний в торговые сигналы
            entries = predictions_series == 2
            exits = predictions_series == 0

        elif task_type == "regression":
            # Динамически определяем стратегию на основе количества таргетов
            num_targets = y_pred_abs.shape[1]
            targets = experiment_cfg.common_params.get("targets", [])

            if num_targets == 1 and targets[0] == 'Close':
                self.log.info("Режим бэктеста: предсказание 1 таргета (Close).")
                # Стратегия: покупаем, если предсказанный Close выше текущего.
                predicted_close = pd.Series(y_pred_abs[:, 0], index=close_prices.index)
                entries = predicted_close > close_prices
                exits = predicted_close < close_prices
            elif num_targets == 2 and 'High' in targets and 'Low' in targets:
                self.log.info("Режим бэктеста: предсказание 2 таргетов (High, Low).")
                # Стратегия: покупаем, если предсказанный Low выше текущего Close.
                pred_high_series = pd.Series(y_pred_abs[:, 0], index=close_prices.index)
                pred_low_series = pd.Series(y_pred_abs[:, 1], index=close_prices.index)
                entries = pred_low_series > close_prices
                exits = pred_high_series < close_prices
            else:
                raise NotImplementedError(
                    f"Логика бэктеста для набора таргетов {targets} (количество: {num_targets}) не реализована."
                )

            ### ### # Извлекаем реальные предсказанные цены в виде pd.Series
            ### pred_high_series = pd.Series(y_pred_abs[:, 0], index=close_prices.index)
            ### pred_low_series = pd.Series(y_pred_abs[:, 1], index=close_prices.index)
            ### 
            ### # Простая стратегия: покупаем, если предсказанный low выше текущего close.
            ### # Продаем, если предсказанный high ниже текущего close.
            ### entries = pred_low_series > close_prices
            ### exits = pred_high_series < close_prices
            
        else:
            raise ValueError(f"Неизвестная логика бэктестинга для типа задачи: {task_type}")

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
        
        financial_metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return_pct": total_return
        }

        # Блок сохранения визуальных артефактов Бэктеста. Сохраняем отчеты, только если есть активный MLflow run
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
            
            # 1. Сохраняем HTML отчет
            html_report_path = self.cfg.ARTIFACTS_DIR / f"backtest_report_{run_id}.html"
            VisualizationUtils.save_backtest_report(portfolio, html_report_path)
            mlflow.log_artifact(str(html_report_path), artifact_path="backtest_reports")
            html_report_path.unlink()
            
            # 2. Сохраняем CSV со сделками
            trades_csv_path = self.cfg.ARTIFACTS_DIR / f"trades_{run_id}.csv"
            VisualizationUtils.save_trades_csv(portfolio, trades_csv_path)
            mlflow.log_artifact(str(trades_csv_path), artifact_path="backtest_reports")
            trades_csv_path.unlink()

        return financial_metrics