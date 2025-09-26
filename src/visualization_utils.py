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
        y_true_abs: pd.DataFrame,
        y_pred_abs: pd.DataFrame,
        save_path: Path,
        full_test_df: pd.DataFrame | None = None,
        enrichment_features_to_plot: list[str] | None = None
    ) -> None:
        """
        Строит и сохраняет график "Предсказания vs. Факт" на основе
        абсолютных, не масштабированных значений.

        Args:
            y_true_abs (pd.DataFrame): DataFrame с истинными абсолютными значениями.
            y_pred_abs (pd.DataFrame): DataFrame с предсказанными абсолютными значениями.
            save_path (Path): Путь для сохранения .png файла.
            full_test_df (pd.DataFrame | None): Полный тестовый датафрейм
                с обогащенными признаками для дополнительной отрисовки.
            enrichment_features_to_plot (list[str] | None): Список имен файлов
                (без расширения), признаки из которых нужно отрисовать.
        """
        log = AppLogger()
        log.info(f"Создание графика 'Предсказания vs. Факт' в '{save_path.name}'...")
        
        try:
            # Строим график
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(15, 7))

            # Отображаем первую цель
            true_col_1_name = y_true_abs.columns[0]
            pred_col_1_name = y_pred_abs.columns[0]
            ax.plot(y_true_abs.index, y_true_abs[true_col_1_name], label=f'Факт ({true_col_1_name})', color='dodgerblue', alpha=0.8)
            ax.plot(y_pred_abs.index, y_pred_abs[pred_col_1_name], label=f'Предсказание ({pred_col_1_name})', color='orangered', linestyle='-')
            
            # Если есть вторая цель (например, Low), отображаем ее тоже
            if len(y_true_abs.columns) > 1:
                true_col_2_name = y_true_abs.columns[1]
                pred_col_2_name = y_pred_abs.columns[1]
                ax.plot(y_true_abs.index, y_true_abs[true_col_2_name], label=f'Факт ({true_col_2_name})', color='deepskyblue', alpha=0.7)
                ax.plot(y_pred_abs.index, y_pred_abs[pred_col_2_name], label=f'Предсказание ({pred_col_2_name})', color='tomato', linestyle='-')                

            ax.set_title('Сравнение предсказаний и фактических значений', fontsize=16)
            ax.set_xlabel('Временные шаги (тестовая выборка)', fontsize=12)
            ax.set_ylabel('Цена', fontsize=12)

            # --- Блок для отрисовки дополнительных признаков ---
            if full_test_df is not None and enrichment_features_to_plot:
                log.info(f"Добавление на график {len(enrichment_features_to_plot)} признаков...")
                ax2 = ax.twinx()  # Создаем вторую ось Y
                ax2.set_ylabel('Значения признаков', fontsize=12)

                # Используем цветовую палитру для разнообразия
                colors = plt.cm.viridis(np.linspace(0, 1, len(enrichment_features_to_plot)))

                ### for i, feature_name_base in enumerate(enrichment_features_to_plot):
                ###     # Формируем префикс для поиска колонок
                ###     prefix_to_find = feature_name_base.lower().replace(" ", "_") + "_"
                ###     
                ###     # Находим все колонки в full_test_df, которые начинаются с этого префикса
                ###     cols_to_plot = [col for col in full_test_df.columns if col.startswith(prefix_to_find)]
                ###     
                ###     if not cols_to_plot:
                ###         log.warning(f"Признаки для '{feature_name_base}' (префикс: '{prefix_to_find}') не найдены в датафрейме.")
                ###         continue
                ### 
                ###     for col_name in cols_to_plot:

                for i, col_name in enumerate(enrichment_features_to_plot):
                    if col_name not in full_test_df.columns:
                        log.warning(f"Колонка '{col_name}' для визуализации не найдена в датафрейме.")
                        continue
                
                    # Убедимся, что данные выравниваются по индексу графика
                    series_to_plot = full_test_df.loc[y_true_abs.index, col_name]

                    ax2.plot(
                        series_to_plot.index,
                        series_to_plot.to_numpy(),
                        label=f'{col_name} (правая ось)',
                        color=colors[i],
                        linestyle=':',
                        alpha=0.7)
                
                # Собираем легенды с обеих осей в одну
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='best')
            else:
                ax.legend() # Если второй оси нет, показываем легенду для первой

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
