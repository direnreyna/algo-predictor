# src/cache_utils.py

import hashlib
import json
from .entities import ExperimentConfig

def get_cache_filename(experiment_cfg: ExperimentConfig, version: str) -> str:
    """
    Генерирует уникальное, но частично читаемое имя файла для кеша.
    Формат: <asset_name>_<feature_set_name>_<version>_<hash>.npz
    """
    # 1. Собираем параметры, влияющие на данные
    params_to_hash = {
        "asset_name": experiment_cfg.asset_name,
        "feature_set_name": experiment_cfg.feature_set_name,
        "labeling_horizon": experiment_cfg.labeling_horizon,
        "task_type": experiment_cfg.task_type,
        # Добавьте сюда любые другие параметры из ExperimentConfig,
        # которые влияют на предобработку данных.
    }
    
    # 2. Создаем стабильную строку из параметров
    params_str = json.dumps(params_to_hash, sort_keys=True).encode('utf-8')
    
    # 3. Генерируем короткий хеш
    hash_str = hashlib.md5(params_str).hexdigest()[:8]
    
    # 4. Собираем читаемое имя файла
    filename = (
        f"{experiment_cfg.asset_name}_"
        f"{experiment_cfg.feature_set_name}_"
        f"{version}_"
        f"{hash_str}.npz"
    )
    return filename