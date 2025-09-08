# src/unpacker.py

import zipfile
import rarfile # Убедитесь, что установлен: pip install rarfile
import tarfile
import py7zr

from pathlib import Path
from .app_config import AppConfig
from .app_logger import AppLogger

# Регистрируем кодек для кириллицы в rarfile, если он нужен
try:
    from unrar import c_unrar
    rarfile.custom_ext_unpack = c_unrar.unpack
except:
    pass

class Unpacker:
    """
    Отвечает за распаковку архивов (.zip, .rar, .tar, .gz, .7z)
    на основе их расширений.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def run(self, file_path: Path) -> bool:
        """
        Проверяет расширение файла и распаковывает его, если это архив.
        Распаковка происходит в ту же папку, где лежит архив.

        :param file_path: Путь к файлу для проверки.
        :return: True, если была произведена распаковка, иначе False.
        """
        lower_filename = file_path.name.lower()
        destination_dir = file_path.parent
        unpacked = False

        if lower_filename.endswith('.zip'):
            unpacked = self._extract_zip(file_path, destination_dir)
        
        elif lower_filename.endswith(('.tar', '.tar.gz', '.tgz')):
            unpacked = self._extract_tar(file_path, destination_dir)

        elif lower_filename.endswith('.7z'):
            unpacked = self._extract_7z(file_path, destination_dir)
            
        elif lower_filename.endswith('.rar'):
            unpacked = self._extract_rar(file_path, destination_dir)

        return unpacked
    

    def _extract_zip(self, path: Path, dest: Path) -> bool:
        self.log.info(f"Обнаружен ZIP-архив: '{path.name}'. Распаковка...")
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(dest)
            self.log.info("Архив успешно распакован.")
            return True
        except zipfile.BadZipFile:
            self.log.error(f"Ошибка чтения ZIP-архива: {path}")
            return False


    def _extract_tar(self, path: Path, dest: Path) -> bool:
        self.log.info(f"Обнаружен TAR/GZ-архив: '{path.name}'. Распаковка...")
        mode = 'r'
        if path.name.lower().endswith(('.tar.gz', '.tgz')):
            mode = 'r:gz'
        try:
            with tarfile.open(path, mode) as tf:
                tf.extractall(dest)
            self.log.info("Архив успешно распакован.")
            return True
        except (tarfile.TarError, EOFError):
            self.log.error(f"Ошибка чтения TAR/GZ-архива: {path}")
            return False
            
    def _extract_7z(self, path: Path, dest: Path) -> bool:
        self.log.info(f"Обнаружен 7z-архив: '{path.name}'. Распаковка...")
        try:
            with py7zr.SevenZipFile(path, mode='r') as z:
                z.extractall(dest)
            self.log.info("Архив успешно распакован.")
            return True
        except Exception as e:
            self.log.error(f"Ошибка чтения 7z-архива: {path}")
            return False

    def _extract_rar(self, path: Path, dest: Path) -> bool:
        self.log.info(f"Обнаружен RAR-архив: '{path.name}'. Распаковка...")
        try:
            with rarfile.RarFile(path) as rf:
                rf.extractall(dest)
            self.log.info("Архив успешно распакован.")
            return True
        except Exception as e:
            self.log.error(f"Ошибка чтения RAR-архива: {path}")
            return False