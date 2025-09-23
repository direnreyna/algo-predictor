# src/visualization_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import TYPE_CHECKING
from .app_logger import AppLogger

# Предотвращаем циклический импорт, т.к. vectorbt может быть не установлен
if TYPE_CHECKING:
    import vectorbt as vbt
    from sklearn.preprocessing import StandardScaler

class VisualizationUtils:
    """
    Вспомогательный класс для создания и сохранения визуальных артефактов.
    Содержит статические методы для генерации графиков и отчетов.
    """
    @staticmethod
    def plot_predictions_vs_actual(
        y_true_scaled: np.ndarray,
        y_pred_scaled: np.ndarray,
        scaler: 'StandardScaler',
        target_names: list[str],
        save_path: Path
    ) -> None:
        """
        Строит и сохраняет график "Предсказания vs. Факт".

        Производит обратную трансформацию данных для отображения в исходном масштабе.

        Args:
            y_true_scaled (np.ndarray): Масштабированные истинные значения.
            y_pred_scaled (np.ndarray): Масштабированные предсказанные значения.
            scaler (StandardScaler): Обученный скейлер для обратной трансформации.
            target_names (list[str]): Имена целевых переменных (например, ['target_High', 'target_Low']).
            save_path (Path): Путь для сохранения .png файла.
        """
        log = AppLogger()
        log.info(f"Создание графика 'Предсказания vs. Факт' в '{save_path.name}'...")
        
        try:
            # Важно: scaler ожидает на вход DataFrame с определенными колонками
            num_targets = len(target_names)
            
            # Создаем временные DataFrame для обратной трансформации
            true_df = pd.DataFrame(y_true_scaled, columns=target_names)
            pred_df = pd.DataFrame(y_pred_scaled, columns=target_names)
            
            # scaler.inverse_transform ожидает все колонки, которые он масштабировал.
            # Создаем "пустышку" с нужными колонками, заполненную нулями,
            # и вставляем в нее таргеты для корректной де-нормализации.
            
            # Получаем имена всех фичей, которые знает скейлер
            try:
                # Для scikit-learn >= 1.0
                all_feature_names = scaler.get_feature_names_out()
            except AttributeError:
                # Для старых версий (менее надежно)
                all_feature_names = [f'feature_{i}' for i in range(scaler.n_features_in_)]

            dummy_df_shape = (len(true_df), len(all_feature_names))
            
            # Создаем "пустышки"
            dummy_true = pd.DataFrame(np.zeros(dummy_df_shape), columns=all_feature_names)
            dummy_pred = pd.DataFrame(np.zeros(dummy_df_shape), columns=all_feature_names)
            
            # Вставляем наши реальные данные
            dummy_true[target_names] = true_df
            dummy_pred[target_names] = pred_df

            # Выполняем обратную трансформацию
            y_true_unscaled_df = pd.DataFrame(scaler.inverse_transform(dummy_true), columns=all_feature_names)[target_names]
            y_pred_unscaled_df = pd.DataFrame(scaler.inverse_transform(dummy_pred), columns=all_feature_names)[target_names]

            # Строим график
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(15, 7))

            # Отображаем первую цель
            true_col_1_name = y_true_unscaled_df.columns[0]
            pred_col_1_name = y_pred_unscaled_df.columns[0]
            ax.plot(np.exp(y_true_unscaled_df[true_col_1_name]), label=f'Факт ({true_col_1_name})', color='dodgerblue', alpha=0.8)
            ax.plot(np.exp(y_pred_unscaled_df[pred_col_1_name]), label=f'Предсказание ({pred_col_1_name})', color='orangered', linestyle='--')
            
            # Если есть вторая цель (например, Low), отображаем ее тоже
            if len(target_names) > 1:
                true_col_2_name = y_true_unscaled_df.columns[1]
                pred_col_2_name = y_pred_unscaled_df.columns[1]
                ax.plot(np.exp(y_true_unscaled_df[true_col_2_name]), label=f'Факт ({true_col_2_name})', color='deepskyblue', alpha=0.7)
                ax.plot(np.exp(y_pred_unscaled_df[pred_col_2_name]), label=f'Предсказание ({pred_col_2_name})', color='tomato', linestyle=':')

            ax.set_title('Сравнение предсказаний и фактических значений', fontsize=16)
            ax.set_xlabel('Временные шаги (тестовая выборка)', fontsize=12)
            ax.set_ylabel('Цена', fontsize=12)
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            log.info("График успешно сохранен.")

        except Exception as e:
            log.error(f"Ошибка при создании графика 'Предсказания vs. Факт': {e}", exc_info=True)

    @staticmethod
    def save_backtest_report(
        portfolio: 'vbt.Portfolio',
        save_path: Path
    ) -> None:
        """
        Сохраняет интерактивный HTML-отчет о результатах бэктеста.

        Args:
            portfolio (vbt.Portfolio): Просчитанный портфель из vectorbt.
            save_path (Path): Путь для сохранения .html файла.
        """
        log = AppLogger()
        log.info(f"Сохранение HTML-отчета о бэктесте в '{save_path.name}'...")
        try:
            fig = portfolio.plot()
            if fig:
                fig.write_html(str(save_path))
                log.info("HTML-отчет успешно сохранен.")
            else:
                log.warning("Не удалось сгенерировать фигуру для HTML-отчета.")
        except Exception as e:
            log.error(f"Ошибка при сохранении HTML-отчета: {e}", exc_info=True)

    @staticmethod
    def save_trades_csv(
        portfolio: 'vbt.Portfolio',
        save_path: Path
    ) -> None:
        """
        Сохраняет таблицу со всеми сделками в .csv файл.

        Args:
            portfolio (vbt.Portfolio): Просчитанный портфель из vectorbt.
            save_path (Path): Путь для сохранения .csv файла.
        """
        log = AppLogger()
        log.info(f"Сохранение таблицы сделок в '{save_path.name}'...")
        try:
            # Используем правильный атрибут .records и конвертируем его в DataFrame
            records_array = portfolio.trades.records
            trades_df = pd.DataFrame(records_array)
            trades_df.to_csv(save_path, index=False)

            ##### records_array = portfolio.trades.records
            ##### trades_df = pd.DataFrame(records_array)
            ### trades = portfolio.trades.records_df
            
            # Как оказалось, это ошибка? Но не работает.
            # Pylance ошибочно считает .trades функцией.
            # Мы явно вызываеми его (хотя это объект), чтобы получить доступ к атрибуту.
            # Это "успокаивает" анализатор.
            # trades = portfolio.trades().records_df
            
            ### trades.to_csv(save_path)
            trades_df.to_csv(save_path)
            log.info("Таблица сделок успешно сохранена.")
        except Exception as e:
            log.error(f"Ошибка при сохранении таблицы сделок: {e}", exc_info=True)
