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

        # Параметры модели
        self.X_LEN = 60
        self.HORIZON = 5
        self.THRESHOLD = 0.01

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