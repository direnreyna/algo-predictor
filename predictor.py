# predictor.py

from src.app_logger import AppLogger
from src.app_config import AppConfig
from src.file_loader import FileLoader
from src.data_preparer import DataPreparer
from src.model_trainer import ModelTrainer
from src.inferencer import Inferencer
from src.entities import ExperimentConfig

class ValuePredictor:
    """
    Основной класс-оркестратор.
    Инициализирует все сервисы и запускает пайплайн обработки данных,
    обучения модели и получения прогноза.
    """
    def __init__(self):
        """Инициализирует классы."""
        self.cfg = AppConfig()
        self.log = AppLogger()
        self.loader = FileLoader(cfg=self.cfg, log=self.log)
        self.preparer = DataPreparer(cfg=self.cfg, log=self.log)
        self.trainer = ModelTrainer(cfg=self.cfg, log=self.log)
        self.inferencer = Inferencer(cfg=self.cfg, log=self.log)

        self.log.separator()
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self):
        """Основной метод, запускающий весь процесс."""
        self.log.info("Начало основного процесса.")
        
        try:
            # 1: Предварительная синхронизация всех источников данных
            self.log.info("Синхронизация источников данных...")
            self.loader.sync_data_sources()
        
            # 2: Подготовка датасета
            # Создаем "ручной" ExperimentConfig на основе глобального AppConfig
            manual_experiment_config = ExperimentConfig(
                asset_name=self.cfg.ASSET_NAME,
                feature_set_name=self.cfg.ACTIVE_FEATURE_SET_NAME,
                labeling_horizon=self.cfg.HORIZON,
                model_type="default_model", # Или берем из AppConfig
                model_params={} # Или берем из AppConfig
            )
            # DataPreparer теперь самодостаточен и сам загрузит нужный файл
            self.preparer.run(experiment_cfg=manual_experiment_config)

          
            # 3: Создание, обучение и оценка модели
            self.trainer.run(experiment_cfg=manual_experiment_config)

            # 4: Предсказание модели
            self.inferencer.run()

            self.log.info("Основной процесс успешно завершен.")

        except Exception as e:
            self.log.critical(f"Критическая ошибка в процессе выполнения: {e}", exc_info=True)
            # exc_info=True добавит полный traceback в лог

if __name__ == '__main__':
    predictor = ValuePredictor()
    predictor.run()