# src/app_config.py
from pathlib import Path

def find_project_root(marker_file=".project-root"):
    """Ищет корневую папку проекта, двигаясь вверх от текущего файла."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent: # Пока не дошли до корня системы
        if (current_dir / marker_file).exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError(f"Не удалось найти корневую папку проекта с маркером '{marker_file}'")

# Определяем корень проекта ОДИН РАЗ
PROJECT_ROOT = find_project_root()
LOCAL_ROOT = Path(__file__).resolve().parent.parent

class AppConfig:
    """
    Класс для хранения всех конфигурационных параметров приложения. Использует паттерн Singleton.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Пути к данным и моделям
        self.PROJECT_ROOT = PROJECT_ROOT
        self.LOCAL_ROOT = LOCAL_ROOT

        self.DATA_DIR = LOCAL_ROOT / "data"
        self.MODELS_DIR = LOCAL_ROOT / "models"
        self.LOGS_DIR = LOCAL_ROOT / "logs"

        ########## ПАРАМЕТРЫ ЗАДАЧИ ##########
        self.TASK_TYPE = "classification" # Варианты: "classification", "regression"
        self.NUM_CLASSES = 3 # Нужно для классификации

        ########## ПАРАМЕТРЫ ЗАГРУЗКИ ##########
        # Имя файла с данными (можно будет передавать извне)
        self.DATA_FILE = "EURUSD_D.csv"
        self.MODEL_FILE = "best_model.keras"

        # Словарь {имя_файла: ссылка_для_скачивания}.
        # Если ссылка None, файл должен лежать в data/ локально.
        self.DOWNLOAD_URLS = {
            "AAPL_D.csv": None,
            "EURRUB_D.csv": None,
            "EURUSD_D.csv": None,
            "GAZP_D.csv": None,
            "SBER_D.csv": None,
            "SBERP_D.csv": None,
            "TSLA_D.csv": None,
            "USDCNH_D.csv": None,
            "USDRUB_D.csv": None
        }

        ########## ПАРАМЕТРЫ ПРЕДОБРАБОТКИ И ЭКСПЕРИМЕНТОВ ##########
        
        # 1. Определение наборов признаков
        # Словарь с базовыми и составными наборами
        _feature_sets = {
            # Базовые "строительные блоки"
            "ohlcv": ['Open', 'Max', 'Min', 'Close', 'Volume'],
            "momentum_indicators": ['RSI_14'],
            "trend_indicators": ['MACDh_12_26_9', 'MACDs_12_26_9'],
            "volatility_indicators": ['BBP_20_2.0', 'ATRr_14'],
            "volume_indicators": ['OBV'],

            # Комбинированные наборы, использующие базовые
            "base_indicators": [
                "ohlcv", "momentum_indicators", "trend_indicators", "volatility_indicators", "volume_indicators"
            ],
            # Кастомный набор (для примера)
            "ohlcv_with_rsi": ["ohlcv", "momentum_indicators"]
        }

        # "Собираем" финальный словарь FEATURE_SETS, раскрывая ссылки
        self.FEATURE_SETS = self._resolve_feature_sets(_feature_sets)

        # 2. Выбор активного эксперимента
        self.ACTIVE_FEATURE_SET_NAME = "base_indicators"
        self.PREPROCESSING_VERSION = "v1.0"
        self.ASSET_NAME = self.DATA_FILE.split('.')[0]

        # 3. Формирование имени файла для кеша
        self.PREPARED_DATA_FILENAME = f"{self.ASSET_NAME}_{self.ACTIVE_FEATURE_SET_NAME}_{self.PREPROCESSING_VERSION}.npz"
        
        # Формируем путь к этому файлу
        self.PREPARED_DATA_PATH = self.DATA_DIR / self.PREPARED_DATA_FILENAME

        # 4. Параметры создания целевой переменной (Labeling)
        self.HORIZON = 5
        self.THRESHOLD = 0.01

        # 5. Параметры разделения на выборки (Splitting)
        self.TEST_SIZE = 250   # Дней (около 1 года)
        self.VAL_SIZE = 250    # Дней
        self.GAP_SIZE = 20     # Разрыв в 20 дней

        ########## ПАРАМЕТРЫ МОДЕЛИ ##########
        self.X_LEN = 60

        # ВЫПОЛНЯЕМ СОЗДАНИЕ ПАПОК ОДИН РАЗ ПРИ ИМПОРТЕ МОДУЛЯ
        self._setup_directories()
        self._initialized = True

    ######################################################################
    def _setup_directories(self):
        """Проверяет и создает все необходимые директории из AppConfig."""
        dirs_to_create = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    ######################################################################
    def _resolve_feature_sets(self, sets: dict) -> dict:
        """
        Раскрывает вложенные ссылки на наборы признаков.
        """
        resolved_sets = {}
        for name, features in sets.items():
            resolved_list = []
            for feature in features:
                # Если элемент - это ссылка на другой набор, раскрываем его
                if feature in sets:
                    resolved_list.extend(sets[feature])
                # Иначе это просто имя признака
                else:
                    resolved_list.append(feature)
            # Убираем дубликаты, сохраняя порядок
            resolved_sets[name] = sorted(list(set(resolved_list)))
        return resolved_sets