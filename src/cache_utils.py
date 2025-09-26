# src/cache_utils.py

import hashlib
import json
from .entities import ExperimentConfig

def get_cache_filename(experiment_cfg: ExperimentConfig, version: str) -> str:
    """
    Генерирует уникальное, но частично читаемое имя файла для кеша.
    Формат: <asset_name>_<feature_set_name>_<version>_<hash>.npz
    """
    common_params = experiment_cfg.common_params
    # 1. Собираем параметры, влияющие на данные
    params_to_hash = {
        "asset_name": common_params.get("asset_name"),
        "feature_set_name": common_params.get("feature_set_name"),
        "labeling_horizon": common_params.get("labeling_horizon"),
        "task_type": common_params.get("task_type"),
        # Добавляем targets в хеш. Преобразуем список в кортеж, чтобы он был хешируемым
        "targets": tuple(common_params.get("targets", [])),
        "enrichment_data": common_params.get("enrichment_data"),
    }
    
    # 2. Создаем стабильную строку из параметров
    params_str = json.dumps(params_to_hash, sort_keys=True).encode('utf-8')
    
    # 3. Генерируем короткий хеш
    hash_str = hashlib.md5(params_str).hexdigest()[:8]
    
    # 4. Собираем читаемое имя файла
    asset_name = common_params.get("asset_name")
    feature_set_name = common_params.get("feature_set_name")
    filename = (
        f"{asset_name}_"
        f"{feature_set_name}_"
        f"{version}_"
        f"{hash_str}.npz"
    )
    return filename