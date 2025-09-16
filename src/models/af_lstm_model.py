# src/models/af_lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import numpy as np

from ..model_factory import BaseModel

class AlgofusionLSTMModel(BaseModel):
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        # Модель будет создана в методе train, когда будет известна форма входа
        self.model = None

    def _build(self, input_shape: tuple, num_outputs: int):
        """
        Строит архитектуру LSTM-модели на основе параметров.
        """
        model = Sequential([
            LSTM(
                units=self.params.get('lstm_units_1', 64),
                return_sequences=True, # Важно для стека из LSTM
                input_shape=input_shape
            ),
            Dropout(self.params.get('dropout_rate', 0.2)),
            LSTM(
                units=self.params.get('lstm_units_2', 32),
                return_sequences=False # Последний LSTM возвращает один вектор
            ),
            Dropout(self.params.get('dropout_rate', 0.2)),
            Dense(num_outputs, activation='linear') # Линейная активация для регрессии
        ])

        learning_rate = self.params.get('learning_rate', 0.001)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        self.model = model

    def train(self, data_dict: dict) -> dict:
        train_dataset = data_dict['train_dataset']
        val_dataset = data_dict.get('val_dataset')

        # Получаем форму входа и количество выходов из первого батча
        sample_x, sample_y = next(iter(train_dataset))
        input_shape = sample_x.shape[1:]  # (timesteps, features)
        num_outputs = sample_y.shape[1]   # количество целевых переменных

        if self.model is None:
            self._build(input_shape, num_outputs)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            train_dataset,
            epochs=100, # Максимальное количество эпох
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0 # Отключаем логирование Keras, т.к. у нас есть свое
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