# src/models/af_lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import numpy as np

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
        if logs:
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.log.info(msg)
        self.epoch_steps = 0 # Сбрасываем счетчик для следующей эпохи

class AlgofusionLSTMModel(BaseModel):
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self.lstm_units_1 = self.params.get('lstm_units_1', 64)
        self.lstm_units_2 = self.params.get('lstm_units_2', 32)
        self.dropout_rate_1 = self.params.get('dropout_rate_1', 0.2)
        self.dropout_rate_2 = self.params.get('dropout_rate_2', 0.4)
        self.out_units = self.params.get('out_units', 1)
        # Модель будет создана в методе train, когда будет известна форма входа
        self.model = None

    def _build_model(self, input_shape: tuple, learning_rate: float):
        """
        Строит и компилирует архитектуру LSTM-модели.
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(self.lstm_units_1, return_sequences=True),
            Dropout(self.dropout_rate_1),
            LSTM(self.lstm_units_2, return_sequences=False),
            Dropout(self.dropout_rate_2),
            Dense(self.out_units, activation='linear')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

    def train(self, data_dict: dict, train_params: dict | None = None) -> dict:
        if train_params is None:
            train_params = {}

        learning_rate = train_params.get('learning_rate', 0.001)
        epochs = train_params.get('epochs', 100)
        es_patience = train_params.get('early_stopping_patience', 10)
        es_monitor = train_params.get('early_stopping_monitor', 'val_loss')

        train_dataset = data_dict.get('train_dataset')
        val_dataset = data_dict.get('val_dataset')

        if train_dataset is None:
            raise ValueError("Для Keras-модели необходим 'train_dataset'.")

        # Получаем форму входа и количество выходов из датасета
        input_shape = train_dataset.element_spec[0].shape[1:]

        if self.model is None:
            self._build_model(input_shape, learning_rate)

        callbacks = [
            EarlyStopping(monitor=es_monitor, patience=es_patience, restore_best_weights=True),
            LoggingCallback()
        ]

        steps_per_epoch = data_dict.get('steps_per_epoch', 1)
        validation_steps = data_dict.get('validation_steps', 1)
        
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=0
        )
        
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите .train() перед .predict()")
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        if self.model:
            self.model.save(path)

    @classmethod
    def load(cls, path: Path) -> 'AlgofusionLSTMModel':
        model_instance = cls({})
        model_instance.model = tf.keras.models.load_model(path)
        return model_instance