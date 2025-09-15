# src/models/lgbm_model.py

import lightgbm as lgb
import joblib
from numpy import ndarray
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor

from ..model_factory import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, model_params:dict):
        super().__init__(model_params)
        # LightGBM сам выберет регрессор или классификатор по параметрам

        base_regressor = lgb.LGBMRegressor(**self.params)
        #=== Предупреждение pylance (про base_regressor) можно и нужно игнорировать. Код является абсолютно корректным и рабочим.
        #=== Это известный и стандартный способ использования lightgbm (и xgboost) внутри экосистемы scikit-learn.
        #=== При запуске кода никакой ошибки не возникнет, MultiOutputRegressor отлично отработает с LGBMRegressor.
        self.model = MultiOutputRegressor(estimator=base_regressor)
        # self.model = lgb.LGBMRegressor(**self.params)

    def train(self, data_dict: dict, train_params: dict | None = None) -> dict:
        if train_params is None: 
            train_params = {}
        # Для LightGBM 'eval_set' является аналогом validation_data
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        ### X_val, y_val = data_dict.get('X_val'), data_dict.get('y_val')

        # для MultiOutputRegressor не применяется eval_set:
        # eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model.fit(X_train, y_train)
        
        # для MultiOutputRegressor не применяются колбэки:
        # self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(10, verbose=False)])
        
        # LightGBM не возвращает историю как Keras, поэтому вернем пустой словарь
        return {}
    
    def predict(self, X):
        return self.model.predict(X)

    def save(self, path:Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path:Path):
        model_instance = cls({}) # Создаем экземпляр с пустыми параметрами
        model_instance.model = joblib.load(path)
        return model_instance

