# src/backtester.py

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
        self.log.info("Запуск финансового бэктеста...")
        
        # --- ЗАГЛУШКА: Эмулируем результаты бэктеста ---
        import random
        financial_metrics = {
            "sharpe_ratio": random.uniform(0.5, 2.5),
            "max_drawdown": -random.uniform(0.05, 0.2)
        }
        # --- КОНЕЦ ЗАГЛУШКИ ---
        
        return financial_metrics