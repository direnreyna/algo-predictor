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

    def read_csv(self, file_name: str, experiment_cfg: ExperimentConfig) -> pd.DataFrame:
        """
        Читает указанный CSV-файл из директории с данными.
        :param file_name: Имя файла (например, "EURUSD_D.csv").
        :param experiment_cfg: Конфигурация эксперимента для доступа к column_mapping.
        :return: DataFrame с данными.
        """
        file_path = self.cfg.DATA_DIR / file_name
        self.log.info(f"Чтение данных из файла '{file_path.name}'...")
        
        if not file_path.exists():
            error_msg = f"Целевой файл '{file_name}' не найден. Выполните sync_data_sources() перед запуском."
            self.log.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # 1. Определяем структуру файла по первой строке ##ДОБАВЛЕН БЛОК
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            has_header = any(char.isalpha() for char in first_line)
            num_columns = len(first_line.split(','))

            # 2. Логика загрузки на основе структуры
            if not has_header:
                self.log.info("Заголовок в файле не обнаружен.")
                if num_columns == 7 and experiment_cfg.column_mapping:
                    self.log.info(f"Обнаружено {num_columns} колонок. Применяем 'column_mapping' из конфига.")
                    col_names = list(experiment_cfg.column_mapping.values())
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

            # 3. Блок обработки datetime
            datetime_col_name = "datetime_temp" # Временное имя колонки
            cols_to_drop = []

            if 'Date' in df.columns and 'Time' in df.columns:
                self.log.info("Обнаружены колонки 'Date' и 'Time'. Объединение в DatetimeIndex.")
                df[datetime_col_name] = df['Date'] + ' ' + df['Time']
                cols_to_drop = ['Date', 'Time']
            elif 'Date' in df.columns:
                self.log.info("Обнаружена колонка 'Date'. Преобразование в DatetimeIndex.")
                df[datetime_col_name] = df['Date']
                cols_to_drop = ['Date']
            
            if cols_to_drop:
                try:
                    df[datetime_col_name] = pd.to_datetime(df[datetime_col_name], format='mixed', errors='coerce')
                    df.set_index(datetime_col_name, inplace=True)
                    df.drop(columns=cols_to_drop, inplace=True)

                    # Удаляем строки, где дата не смогла быть преобразована
                    df = df[df.index.notna()]
                    self.log.info("DatetimeIndex успешно создан и установлен.")
                except Exception as e:
                    self.log.error(f"Критическая ошибка при создании DatetimeIndex: {e}")
                    raise
            else:
                self.log.warning("Колонки 'Date'/'Time' не найдены. Данные не будут индексированы по времени.")

            self.log.info(f"Данные успешно загружены. Размер: {df.shape}")
            ### self.log.info(f"Первые 5 строк обработанных данных из FileLoader:\n{df.head().to_string()}")
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