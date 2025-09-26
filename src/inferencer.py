# src.inferencer.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from mlflow import MlflowClient

from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig
from .model_factory import ModelFactory
from .file_loader import FileLoader
from .feature_engineer import FeatureEngineer
from .statistical_analyzer import StatisticalAnalyzer
from .dataset_builder import DatasetBuilder
from .inverse_transformer import InverseTransformer

class Inferencer:

    """
    Отвечает за применение обученной модели к новым, ранее не виданным данным.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.client = MlflowClient()
        # Инициализируем все необходимые инструменты для предобработки
        self.file_loader = FileLoader(cfg, log)
        self.feature_engineer = FeatureEngineer(cfg, log)
        self.statistical_analyzer = StatisticalAnalyzer(cfg, log)
        self.dataset_builder = DatasetBuilder(cfg, log)
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, run_id: str, data_path: Path) -> pd.DataFrame:
        """
        Выполняет полный пайплайн инференса для указанного запуска и новых данных.

        Args:
            run_id (str): ID запуска в MLflow, из которого будут загружены артефакты.
            data_path (Path): Путь к новому CSV-файлу с данными для предсказания.

        Returns:
            pd.DataFrame: DataFrame с предсказаниями в абсолютных ценах,
                          индексированный по дате.
        """
        self.log.info(f"--- Начало инференса для MLflow Run ID: {run_id} ---")

        # 1. Загрузка метаданных и артефактов из MLflow
        self.log.info("Загрузка артефактов и конфигурации из MLflow...")
        run_data = self.client.get_run(run_id).data
        
        # Восстанавливаем конфиг эксперимента из параметров MLflow
        # (Простое восстановление, можно усложнить для вложенных словарей при необходимости)
        params = run_data.params
        common_params = {k: v for k, v in params.items() if k in self.cfg.DOWNLOAD_URLS.keys() or k in ["asset_name", "model_type", "feature_set_name", "differencing", "task_type", "targets", "labeling_horizon", "x_len"]}
        common_params['targets'] = eval(params['targets']) # Преобразуем строку '["High", "Low"]' в список
        
        experiment_cfg = ExperimentConfig(common_params=common_params)
        experiment_cfg.was_differenced = params.get('was_differenced', 'False').lower() == 'true'

        # Загружаем модель и скейлер
        local_path = self.client.download_artifacts(run_id=run_id, path="")
        artifact_path = Path(local_path)
        
        model_file = next((artifact_path / "model").iterdir())
        model_object = ModelFactory.get_model(common_params['model_type'], {}).load(model_file)
        
        # Динамически получаем имя файла скейлера из тегов
        scaler_filename = run_data.tags.get("scaler_filename")
        if not scaler_filename:
            raise ValueError(f"Тег 'scaler_filename' не найден в запуске {run_id}. Инференс невозможен.")

        scaler_file = artifact_path / "preprocessor" / scaler_filename        
        scaler = joblib.load(scaler_file)
        self.log.info("Артефакты (модель, скейлер) и конфиг успешно загружены.")

        # 2. Предобработка новых данных
        self.log.info(f"Обработка новых данных из файла: {data_path.name}")
        df = self.file_loader.read_csv(data_path.name, experiment_cfg=experiment_cfg)
        original_df = df.copy() # Сохраняем для InverseTransformer
        
        df_featured = self.feature_engineer.run(df, experiment_cfg)
        df_transformed, _ = self.statistical_analyzer.transform(df_featured, experiment_cfg)
        
        cols_to_scale = df_transformed.select_dtypes(include='number').columns.tolist()
        df_transformed[cols_to_scale] = scaler.transform(df_transformed[cols_to_scale])
        self.log.info("Предобработка новых данных завершена.")
        
        # 3. Подготовка данных для модели
        # Для инференса нам не нужны y, поэтому передаем пустой список target_cols
        data_dict = self.dataset_builder.build(
            model_type=common_params['model_type'],
            datasets={'inference': df_transformed.to_numpy()},
            target_cols=[],
            all_cols=df_transformed.columns.tolist(),
            experiment_cfg=experiment_cfg
        )
        X_inference = data_dict['X_inference']

        # 4. Получение предсказаний
        self.log.info("Получение предсказаний от модели...")
        predictions = model_object.predict(X_inference)
        if not isinstance(predictions, np.ndarray):
            try:
                # Попытка преобразовать sparse matrix в плотный массив
                predictions = predictions.toarray()
            except AttributeError:
                # Общий случай для других не-numpy типов
                predictions = np.array(predictions)

        # 5. Обратное преобразование
        self.log.info("Выполнение обратного преобразования предсказаний...")
        inverse_transformer = InverseTransformer(
            scaler=scaler,
            original_test_df=original_df,
            all_cols=df_transformed.columns.tolist(),
            experiment_cfg=experiment_cfg
        )
        y_pred_abs = inverse_transformer.transform(predictions)
        
        # 6. Форматирование результата
        x_len = int(common_params.get("x_len", 22))
        # Предсказания начинаются с `x_len`-го элемента, т.к. первые окна ушли на формирование первого X
        pred_index = original_df.index[x_len:]
        
        result_df = pd.DataFrame(y_pred_abs, index=pred_index, columns=common_params['targets'])
        self.log.info(f"--- Инференс успешно завершен. Получено {len(result_df)} предсказаний. ---")
        
        return result_df