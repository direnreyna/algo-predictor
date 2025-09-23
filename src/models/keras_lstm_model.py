# src/models/keras_lstm_model.py

import tensorflow as tf
from numpy import ndarray
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from ..model_factory import BaseModel
from ..app_logger import AppLogger

class LoggingCallback(tf.keras.callbacks.Callback):
    """Кастомный callback для вывода логов обучения через наш логгер."""
    def __init__(self):
        super().__init__()
        self.log = AppLogger()
        self.epoch_steps = 0

    def on_train_batch_end(self, batch, logs=None):
        """Подсчитывает количество шагов в эпохе."""
        self.epoch_steps += 1

    def on_epoch_end(self, epoch, logs=None):
        msg = f"Эпоха {epoch + 1} ({self.epoch_steps} шагов) | "        
        ## msg = f"Эпоха {epoch + 1} | "
        if logs:
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.log.info(msg)
        self.epoch_steps = 0 # Сбрасываем счетчик для следующей эпохи

class KerasLSTMModel(BaseModel):
    """
    Реализация модели на базе LSTM с использованием TensorFlow/Keras.
    """
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        # Параметры с дефолтными значениями для отказоустойчивости
        self.lstm_units_1 = self.params.get('lstm_units_1', 64)
        self.lstm_units_2 = self.params.get('lstm_units_2', 32)
        self.dropout_rate_1 = self.params.get('dropout_rate_1', 0.2)
        self.dropout_rate_2 = self.params.get('dropout_rate_2', 0.4)
        self.dense_units = self.params.get('dense_units', 32)
        self.out_units = self.params.get('out_units', 2)
        # Модель не создается здесь, а будет создана в методе train,
        self.model = None

    def _build_model(self, input_shape, learning_rate):
        """Приватный метод для построения архитектуры модели."""
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(self.lstm_units_1, return_sequences=True),
            Dropout(self.dropout_rate_1),
            LSTM(self.lstm_units_2),
            Dropout(self.dropout_rate_2),
            Dense(self.dense_units, activation='relu'),
            Dense(self.out_units, activation='linear') # 2 выхода: high и low
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

    def train(self, data_dict: dict, train_params: dict | None = None) -> dict:
        """
        Обучает LSTM-модель, используя tf.data.Dataset.
        """

        if train_params is None:
            train_params = {}

        # Извлекаем параметры обучения
        learning_rate = train_params.get('learning_rate', 0.001)
        epochs = train_params.get('epochs', 50)
        es_patience = train_params.get('early_stopping_patience', 10)
        es_monitor = train_params.get('early_stopping_monitor', 'val_loss')

        train_dataset = data_dict.get('train_dataset')
        val_dataset = data_dict.get('val_dataset')

        if train_dataset is None:
            raise ValueError("Для Keras-модели необходим 'train_dataset'.")
        
        # Получаем shape из первого батча данных и строим модель
        if self.model is None:
            input_shape = train_dataset.element_spec[0].shape[1:]
            self._build_model(input_shape, learning_rate=learning_rate)

        callbacks = [
            EarlyStopping(monitor=es_monitor, patience=es_patience, restore_best_weights=True),
            LoggingCallback()
        ]

        # Получаем рассчитанные шаги из DatasetBuilder, с дефолтом на случай отсутствия
        steps_per_epoch = data_dict.get('steps_per_epoch', 1)
        validation_steps = data_dict.get('validation_steps', 1)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=0 # Отключаем вывод Keras в консоль, т.к. у нас свой логгер
        )
        
        return history.history

    def predict(self, X: ndarray) -> ndarray:
        if self.model is None:
            raise ValueError("Попытка сделать предсказание на необученной модели Keras. Атрибут 'self.model' равен None.")
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        # Keras 2.9+ предпочитает формат .keras
        # path_with_ext = path.with_suffix('.keras')
        if self.model is None:
            raise ValueError("Попытка сохранить необученную модель Keras. Атрибут 'self.model' равен None.")
        self.model.save(path)

    @classmethod
    def load(cls, path: Path) -> 'KerasLSTMModel':
        if not path.exists():
            raise FileNotFoundError(f"Файл модели Keras не найден по пути: {path}")
            
        try:
            loaded_keras_model = tf.keras.models.load_model(path)
        except Exception as e:
            raise IOError(f"Не удалось загрузить модель Keras из файла '{path.name}'. Исходная ошибка: {e}")
                
        # Создаем экземпляр нашего класса и "прикрепляем" к нему загруженную модель
        model_instance = cls({}) 
        model_instance.model = loaded_keras_model
        return model_instance