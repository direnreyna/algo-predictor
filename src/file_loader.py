# src.file_loader.py

import os
import pandas as pd
from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig
from .downloader import Downloader
from .unpacker import Unpacker
from pathlib import Path

class FileLoader:
    """
    Отвечает за подготовку и чтение исходных файлов данных. 
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.downloader = Downloader(cfg, log)
        self.unpacker = Unpacker(cfg, log)
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def sync_data_sources(self) -> None:
        """
        Гарантирует наличие всех необходимых CSV-файлов на диске.
        Выполняет скачивание и распаковку всех источников, указанных в конфиге.
        Этот метод вызывается один раз перед серией экспериментов.
        """
        self.log.info("--- Начало синхронизации источников данных ---")
        
        # 1: Скачиваем все файлы, для которых есть URL
        self._download_all_sources()
    
        # 2: Ищем и распаковываем все архивы в папке data/
        self._unpack_all_archives()
        
        self.log.info("--- Синхронизация источников данных завершена ---")

    def read_csv(self, file_name: str, experiment_cfg: ExperimentConfig = None, data_type: str = None) -> pd.DataFrame:
        """
        Читает CSV-файл. Может работать в двух режимах:
        1. Для основных данных (experiment_cfg != None): использует column_mapping.
        2. Для внешних данных (data_type != None): использует EXTERNAL_DATA_SCHEMAS.

        Args:
            file_name (str): Имя файла.
            experiment_cfg (ExperimentConfig, optional): Конфиг эксперимента.
            data_type (str, optional): Тип внешних данных ('interest_rate', 'cpi').

        Returns:
            pd.DataFrame: Загруженный и обработанный DataFrame.
        """
        file_path = self.cfg.DATA_DIR / file_name
        self.log.info(f"Чтение данных из файла '{file_path.name}'...")
        
        if not file_path.exists():
            error_msg = f"Целевой файл '{file_name}' не найден. Выполните sync_data_sources() перед запуском."
            self.log.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:

            # --- РЕЖИМ 2: Загрузка внешних данных по схеме ---
            if data_type:
                schema = self.cfg.EXTERNAL_DATA_SCHEMAS.get(data_type)
                if not schema:
                    raise ValueError(f"Схема для типа данных '{data_type}' не найдена в AppConfig.")

                # Валидация заголовка
                with open(file_path, 'r', encoding='utf-8') as f:
                    header_line = f.readline().strip()
                
                expected_header = schema['separator'].join(schema['columns'])
                if header_line != expected_header:
                    raise ValueError(
                        f"Ошибка валидации заголовка в файле '{file_name}'.\n"
                        f"  Ожидалось: '{expected_header}'\n"
                        f"  Получено:  '{header_line}'"
                    )
                
                # Загрузка данных
                df = pd.read_csv(file_path, sep=schema['separator'], decimal=',')
                
                # Присваиваем префикс, чтобы избежать конфликта имен колонок
                prefix = file_path.stem.lower().replace(" ", "_") + "_"
                df = df.rename(columns={col: f"{prefix}{col}" for col in df.columns if col != 'Date'})

            # --- РЕЖИМ 1: Загрузка основных данных эксперимента ---
            else:
                if experiment_cfg is None:
                    raise ValueError("experiment_cfg обязателен при загрузке основных данных (когда data_type is None).")

                # 1. Определяем структуру файла по первой строке
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                has_header = any(char.isalpha() for char in first_line)
                num_columns = len(first_line.split(','))

                # 2. Логика загрузки на основе структуры
                if not has_header:
                    self.log.info("Заголовок в файле не обнаружен.")
                    column_mapping = experiment_cfg.common_params.get("column_mapping")
                    if num_columns == 7 and column_mapping:
                        self.log.info(f"Обнаружено {num_columns} колонок. Применяем 'column_mapping' из конфига.")
                        col_names = list(column_mapping.values())
                        df = pd.read_csv(file_path, header=None, names=col_names)
                    else:
                        raise ValueError(
                            f"Файл '{file_name}' не содержит заголовка и имеет {num_columns} колонок (ожидалось 7) "
                            f"или 'column_mapping' не предоставлен в конфиге. Автоматическая загрузка невозможна."
                        )
                else: # has_header is True
                    self.log.info("Обнаружена строка заголовка. Загрузка данных со стандартной обработкой.")
                    df = pd.read_csv(file_path)

                    # Стандартизация имен колонок
                    df_columns_lower = {col.lower().strip(): col for col in df.columns}
                    rename_map = {}
                    possible_names = {
                        'Date': ['date', '<date>'], 'Time': ['time', '<time>'],
                        'Open': ['open', '<open>'], 'High': ['high', 'max', '<high>'],
                        'Low': ['low', 'min', '<low>'], 'Close': ['close', '<close>'],
                        'Volume': ['volume', 'vol', '<vol>']
                    }
                    for standard_name, variants in possible_names.items():
                        for variant in variants:
                            if variant in df_columns_lower:
                                rename_map[df_columns_lower[variant]] = standard_name
                                break
                    df.rename(columns=rename_map, inplace=True)

            # Обработка datetime
            if 'Date' in df.columns:
                date_series = df['Date']
                cols_to_drop = ['Date']

                if 'Time' in df.columns:
                    self.log.info("Объединение 'Date' и 'Time' в DatetimeIndex.")
                    date_series = df['Date'] + ' ' + df['Time']
                    cols_to_drop.append('Time')
                else:
                    self.log.info("Преобразование колонки 'Date' в DatetimeIndex.")

                df_with_dt_index = df.set_index(pd.to_datetime(date_series, dayfirst=True, errors='coerce'))
                df_with_dt_index.drop(columns=cols_to_drop, inplace=True)
                df_with_dt_index = df_with_dt_index[df_with_dt_index.index.notna()]
                df_with_dt_index.sort_index(inplace=True)
                self.log.info("DatetimeIndex успешно создан и установлен.")
                df = df_with_dt_index
            else:
                self.log.warning(f"Колонка 'Date' не найдена в '{file_name}'. Данные не будут индексированы по времени.")

            self.log.info(f"Данные успешно загружены. Размер: {df.shape}")
            return df

        except Exception as e:
            self.log.error(f"Ошибка при чтении файла '{file_name}': {e}")
            raise

    def _download_all_sources(self):
        """Скачивает все файлы, указанные в DOWNLOAD_URLS."""
        self.log.info("Проверка и скачивание исходных файлов...")
        for filename, url in self.cfg.DOWNLOAD_URLS.items():
            if url:
                filepath = self.cfg.DATA_DIR / filename
                self.downloader.run(url, filepath)

    def _unpack_all_archives(self):
        """Ищет все архивы в папке data/, распаковывает их и удаляет исходные архивы."""
        self.log.info("Поиск, распаковка и очистка архивов...")
        # Создаем копию списка, так как будем изменять содержимое папки в цикле
        for filename in list(os.listdir(self.cfg.DATA_DIR)):
            filepath = self.cfg.DATA_DIR / filename
            if filepath.is_file():
                unpacked = self.unpacker.run(filepath)
                if unpacked:
                    # Если распаковка была успешной, удаляем исходный архив
                    try:
                        os.remove(filepath)
                        self.log.info(f"Исходный архив '{filename}' удален.")
                    except OSError as e:
                        self.log.error(f"Не удалось удалить архив '{filename}': {e}")