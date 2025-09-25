# src.model_trainer.py

import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import joblib
import json
from mlflow import MlflowClient
from pathlib import Path

from .app_config import AppConfig
from .app_logger import AppLogger
from .data_saver import DataSaver        
from .entities import ExperimentConfig
from .model_factory import ModelFactory
from .dataset_builder import DatasetBuilder
from .metrics_calculator import MetricsCalculator
from .visualization_utils import VisualizationUtils
from .inverse_transformer import InverseTransformer
from .cache_utils import get_cache_filename

class ModelTrainer:
    """
    Отвечает за создание, обучение и сохранение модели.

    Нюанс:
    Преобразование в OHE — это задача ModelTrainer, потому что это техническое требование модели (loss='categorical_crossentropy'), 
    а не часть семантической подготовки данных. И самый лучший способ сделать это — "на лету" с помощью tf.data.Dataset.map()
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.data_saver = DataSaver(cfg, log)
        self.model_factory = ModelFactory()
        self.dataset_builder = DatasetBuilder(cfg, log)
        self.metrics_calculator = MetricsCalculator()
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, experiment_cfg: ExperimentConfig, warm_start: dict | None = None, log_history_per_epoch: bool = False) -> dict:
        """Запускает полный цикл обучения модели, оркестрируя другие сервисы.

        Args:
            experiment_cfg (ExperimentConfig): Конфигурация текущего эксперимента.
            warm_start (dict | None): Конфигурация для дообучения. 
                                      Ожидает {'run_id': '...'}. По умолчанию None.
            log_history_per_epoch (bool): Флаг, включающий логирование метрик на каждой эпохе.

        Returns:
            dict: Словарь с результатами: мл-метрики, предсказания,
                  тестовые данные и скейлер для бэктестера.
        """
        common_params = experiment_cfg.common_params
        model_type = common_params.get("model_type")
        task_type = common_params.get("task_type")
        if not model_type or not task_type:
            raise ValueError("В common_params отсутствуют обязательные ключи 'model_type' или 'task_type'.")

        self.log.info(f"Запуск процесса обучения для модели типа '{model_type}'...")

        # 1. Загрузка данных и артефактов
        cache_filename = get_cache_filename(experiment_cfg, self.cfg.PREPROCESSING_VERSION)
        cache_path = self.cfg.DATA_DIR / cache_filename
        
        datasets = self.data_saver.load(file_path=cache_path)
        original_test_df_raw = datasets['original_test']
        
        scaler_path = cache_path.with_suffix('.joblib')
        scaler = joblib.load(scaler_path)
        
        metadata_path = cache_path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        all_cols = metadata['columns']
        target_cols = metadata['target_columns']

        original_test_df = pd.DataFrame(original_test_df_raw, columns=metadata['columns'])

        # Создаем экземпляр InverseTransformer, который инкапсулирует всю логику
        inverse_transformer = InverseTransformer(
            scaler=scaler,
            original_test_df=original_test_df,
            all_cols=all_cols,
            experiment_cfg=experiment_cfg
        )

        # 2. Подготовка данных (делегирование DatasetBuilder)
        data_dict = self.dataset_builder.build(
            model_type=model_type,
            datasets=datasets,
            target_cols=target_cols,
            all_cols=all_cols
        )
        
        # 3. Создание или загрузка модели
        if warm_start and warm_start.get("run_id"):
            source_run_id = warm_start["run_id"]
            self.log.info(f"Режим дообучения. Загрузка модели из MLflow run_id: {source_run_id}")
            try:
                client = MlflowClient()
                local_path = client.download_artifacts(run_id=source_run_id, path="model")

                # Имя файла внутри папки 'model' может быть разным
                model_file = next(Path(local_path).iterdir())
                
                # Используем нашу фабрику для загрузки, т.к. она знает о BaseModel
                model_object = ModelFactory.get_model(model_type, {}).load(model_file)
                self.log.info("Модель для дообучения успешно загружена.")
            except Exception as e:
                self.log.error(f"Не удалось загрузить модель из run_id '{source_run_id}': {e}")
                raise        
        else:        
            # 3. Создание модели (делегирование ModelFactory)
            model_object = self.model_factory.get_model(
                model_type=model_type,
                model_params=experiment_cfg.model_params
            )
        
        # 4. Обучение модели (делегирование объекту модели)
        self.log.info("Начало обучения модели...")
        ###history = model_object.train(data_dict) # Передаем весь словарь
        history = model_object.train(data_dict, train_params=experiment_cfg.train_params) # Передаем весь словарь 
        self.log.info("Обучение модели завершено.")

        # 5. Оценка модели
        predictions = model_object.predict(data_dict['X_test'])
        if not isinstance(predictions, np.ndarray):
            try:
                predictions = predictions.toarray() # для sparse matrix
            except AttributeError:
                predictions = np.array(predictions) # для других типов

        # Получаем абсолютные значения для корректного расчета метрик
        y_pred_abs = inverse_transformer.transform(predictions)

        # Извлекаем истинные АБСОЛЮТНЫЕ значения из оригинального датасета
        num_predictions = len(predictions)
        offset = len(original_test_df) - num_predictions
        target_names = experiment_cfg.common_params.get("targets", [])
        
        y_true_abs_list = []
        for name in target_names:
            true_values = original_test_df[name].iloc[offset:].values
            y_true_abs_list.append(true_values)
        y_true_abs = np.stack(y_true_abs_list, axis=1)

        ml_metrics = self.metrics_calculator.calculate(
            task_type=task_type,
            y_true_abs=y_true_abs,
            y_pred_abs=y_pred_abs
        )

        # Добавляем метрики из истории обучения, если они есть
        if history:
            # Логируем историю обучения в MLflow ПОШАГОВО для красивых графиков
            if log_history_per_epoch:
                self.log.info("Детальное логирование истории обучения включено.")
                for metric_name, values in history.items():
                    for step, value in enumerate(values):
                        mlflow.log_metric(f"history_{metric_name}", value, step=step)

            for key, value in history.items():
                # Берем последнее значение из списка эпох для возврата
                if isinstance(value, list) and value:
                    ml_metrics[f'history_{key}'] = value[-1]
        
        # 6. Сохранение и логирование артефактов в MLflow
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
            
            # Создаем DataFrame с правильным индексом для графика
            plot_index = original_test_df.index[offset:]
            y_true_df = pd.DataFrame(y_true_abs, index=plot_index, columns=target_names)
            y_pred_df = pd.DataFrame(y_pred_abs, index=plot_index, columns=target_names)
            
            # Создаем и логируем график "Предсказания vs. Факт"
            chart_path = self.cfg.ARTIFACTS_DIR / f"predictions_vs_actual_{run_id}.png"
            VisualizationUtils.plot_predictions_vs_actual(
                y_true_abs=y_true_df,
                y_pred_abs=y_pred_df,
                save_path=chart_path
            )
            mlflow.log_artifact(str(chart_path), artifact_path="charts")
            chart_path.unlink() # Удаляем временный файл
            
            # Определяем расширение файла в зависимости от типа модели
            model_ext = ".keras" if model_type in ["lstm", "af_lstm"] else ".pkl"

            # Сохраняем модель во временный файл, чтобы залогировать в MLflow
            # temp_model_path = Path(self.cfg.MODELS_DIR) / f"temp_model_{run_id}.pkl"     
            temp_model_path = Path(self.cfg.MODELS_DIR) / f"temp_model_{run_id}{model_ext}"

            model_object.save(temp_model_path)
            
            mlflow.log_artifact(str(temp_model_path), artifact_path="model")
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessor")

            # Удаляем временный файл модели
            temp_model_path.unlink()
        else:
            self.log.warning("Нет активного MLflow run. Артефакты не будут залогированы.")

        # Определяем имя ключевой метрики и выводим его
        key_metric_name = "mse" if task_type == "regression" else "accuracy"

        key_metric_value = ml_metrics.get(key_metric_name, 0.0)
        self.log.info(f"Оценка модели завершена. Ключевая метрика ({key_metric_name}): {key_metric_value:.4f}")

        # 7. Подготовка тестовых данных для бэктестера
        # Нам нужен исходный, но уже масштабированный DataFrame `test`,
        # а не нарезанный на окна X_test/y_test.
        # Загрузим его из кеша еще раз, но уже как pandas DataFrame.
        test_data_raw = datasets['test']
        test_df = pd.DataFrame(test_data_raw, columns=all_cols)

        return {
            "ml_metrics": ml_metrics,
            "predictions": predictions,
            "test_df": test_df,
            "inverse_transformer": inverse_transformer
        }
