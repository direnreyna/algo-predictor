# src/models/keras_lstm_model.py

import tensorflow as tf
from numpy import ndarray
from pathlib import Path

from ..model_factory import BaseModel
from ..app_logger import AppLogger

class LoggingCallback(tf.keras.callbacks.Callback):
    """Кастомный callback для вывода логов обучения через наш логгер."""
    def __init__(self):
        super().__init__()
        self.log = AppLogger()

    def on_epoch_end(self, epoch, logs=None):
        msg = f"Эпоха {epoch+1:03d} | "
        if logs:
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.log.info(msg)

class KerasLSTMModel(BaseModel):
    """
    Реализация модели на базе LSTM с использованием TensorFlow/Keras.
    """
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        # Параметры с дефолтными значениями для отказоустойчивости
        self.lstm_units = self.params.get('lstm_units', 64)
        self.dropout_rate = self.params.get('dropout_rate', 0.2)
        ### self.learning_rate = self.params.get('learning_rate', 0.001)
        ### self.epochs = self.params.get('epochs', 50)
        
        # Модель не создается здесь, а будет создана в методе train,
        # так как для этого нужен shape входных данных.
        self.model = None

    def _build_model(self, input_shape, learning_rate):
        """Приватный метод для построения архитектуры модели."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, name="input"),
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.LSTM(self.lstm_units // 2),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear', name="output") # 2 выхода: high и low
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

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
        batch_size = train_params.get('batch_size', 32)

        train_dataset = data_dict.get('train_dataset')
        val_dataset = data_dict.get('val_dataset')
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        if train_dataset is None:
            raise ValueError("Для Keras-модели необходим 'train_dataset'.")
        
        # Получаем shape из первого батча данных и строим модель
        if self.model is None:
            # (timesteps, features)
            # input_shape = next(iter(train_dataset))[0].shape[1:]
            input_shape = train_dataset.element_spec[0].shape[1:]
            self._build_model(input_shape, learning_rate)


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            restore_best_weights=True
        )

        # Рассчитываем шаги для Keras, чтобы он знал, когда заканчивается эпоха
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        # Убедимся, что шаги не равны нулю, если данных мало
        if steps_per_epoch == 0: steps_per_epoch = 1
        if validation_steps == 0: validation_steps = 1

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping, LoggingCallback()],
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