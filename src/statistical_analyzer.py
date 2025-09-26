# src/statistical_analyzer.py

import pandas as pd
import numpy as np
import mlflow
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from .app_config import AppConfig
from .app_logger import AppLogger
from .entities import ExperimentConfig

class StatisticalAnalyzer:
    """
    Отвечает за статистический анализ и трансформацию временных рядов
    для приведения их к стационарному виду и обработки артефактов.
    """
    def __init__(self, cfg: AppConfig, log: AppLogger):
        """
        Инициализирует анализатор.

        Args:
            cfg (AppConfig): Глобальный конфигурационный объект.
            log (AppLogger): Экземпляр логгера.
        """
        self.cfg = cfg
        self.log = log
        self.log.info(f"Класс {self.__class__.__name__} инициализирован.")

    def analyze(self, df: pd.DataFrame, experiment_cfg: 'ExperimentConfig') -> dict:
        """
        Выполняет диагностику временного ряда (Фаза 1).
        Анализирует ВЕСЬ DataFrame на предмет стационарности, тренда и т.д.,
        используя первую целевую переменную как репрезентативный ряд.

        Args:
            df (pd.DataFrame): Исходный DataFrame с ценами.
            experiment_cfg (ExperimentConfig): Конфигурация для определения
                целевой переменной для анализа.

        Returns:
            dict: Словарь с результатами анализа (global_insights).
        """
        self.log.info("Фаза 1: Запуск статистической диагностики...")
        
        targets = experiment_cfg.common_params.get("targets", [])
        if not targets:
            raise ValueError("Список 'targets' не найден в common_params конфига.")

        representative_series_name = targets[0]
        series_to_analyze = df[representative_series_name]
        self.log.info(f"Анализ будет проводиться по репрезентативному ряду: '{representative_series_name}'")
        
        ### close_series = df['Close']
        
        # 1. Тест на стационарность
        adf_result = adfuller(series_to_analyze.dropna())
        is_stationary = adf_result[1] < 0.05 # p-value < 0.05
        
        log_message = f"  ADF-тест на стационарность (p-value): {adf_result[1]:.4f}. Стационарный: {is_stationary}"
        if not is_stationary:
            self.log.warning(log_message + " -> Ряд НЕ стационарен, что может ухудшить качество моделей.")
        else:
            self.log.info(log_message)

        # 2. Расчет показателя Хёрста
        hurst_exponent = self._calculate_hurst(series_to_analyze)
        hurst_interpretation = "Не рассчитан"
        if not np.isnan(hurst_exponent):
            if hurst_exponent > 0.5: hurst_interpretation = "Трендовый"
            elif hurst_exponent < 0.5: hurst_interpretation = "Возврат к среднему"
            else: hurst_interpretation = "Случайное блуждание"
            self.log.info(f"  Показатель Хёрста: {hurst_exponent:.4f} ({hurst_interpretation})")
        else:
            self.log.warning("  Не удалось рассчитать показатель Хёрста (слишком короткий ряд).")

        # 3. Тест на автокорреляцию (Льюнга-Бокса) на доходностях
        returns = series_to_analyze.pct_change().dropna()
        ljung_box_result = acorr_ljungbox(returns, lags=[21], return_df=True)
        lb_pvalue = ljung_box_result['lb_pvalue'].iloc[0]
        has_autocorrelation = lb_pvalue < 0.05

        log_message = f"  Тест Льюнга-Бокса на автокорреляцию (p-value): {lb_pvalue:.4f}. Автокорреляция: {has_autocorrelation}"
        if has_autocorrelation:
            self.log.warning(log_message + " -> Обнаружена значимая автокорреляция в доходностях.")
        else:
            self.log.info(log_message)

        # 4. Анализ "толстых хвостов" (Эксцесс)
        kurtosis_value = returns.kurtosis() # Расчет эксцесса (избыточный эксцесс)
        kurtosis_interpretation = "Не рассчитан"
        try:
            kurtosis_value = float(str(kurtosis_value))
            if kurtosis_value > 0:
                kurtosis_interpretation = "Лептокуртическое (толстые хвосты)"
                self.log.warning(f"  Эксцесс доходностей: {kurtosis_value:.4f} ({kurtosis_interpretation}) -> Присутствуют 'толстые хвосты'.")
            else:
                kurtosis_interpretation = "Платикуртическое (тонкие хвосты)" if kurtosis_value < 0 else "Мезокуртическое (нормальное распределение)"
                self.log.info(f"  Эксцесс доходностей: {kurtosis_value:.4f} ({kurtosis_interpretation})")
        except:
            self.log.warning(f"ПРЕДУПРЕЖДЕНИЕ: Эксцесс доходностей: {kurtosis_value} - не является числом!")    

        # 5. Тест на гетероскедастичность (ARCH-эффект)
        arch_test_result = het_arch(returns)
        arch_pvalue = arch_test_result[1]
        has_arch_effect = arch_pvalue < 0.05

        log_message = f"  ARCH-тест на гетероскедастичность (p-value): {arch_pvalue:.4f}. ARCH-эффект: {has_arch_effect}"
        if has_arch_effect:
            self.log.warning(log_message + " -> Обнаружен ARCH-эффект (кластеризация волатильности).")
        else:
            self.log.info(log_message)

        # 6. Формирование и логирование статистического заключения
        summary_text = (
            f"===== Статистическое заключение по ряду '{series_to_analyze.name}' =====\n"
            f"1. Стационарность (ADF-тест):\n"
            f"   - p-value: {adf_result[1]:.4f}\n"
            f"   - Вывод: {'Стационарный' if is_stationary else 'НЕ СТАЦИОНАРНЫЙ'}\n\n"
            f"2. Трендовость (Показатель Хёрста):\n"
            f"   - Значение: {hurst_exponent:.4f}\n"
            f"   - Вывод: {hurst_interpretation}\n\n"
            f"3. Автокорреляция доходностей (Ljung-Box):\n"
            f"   - p-value: {lb_pvalue:.4f}\n"
            f"   - Вывод: {'Присутствует' if has_autocorrelation else 'Отсутствует'}\n\n"
            f"4. Форма распределения (Эксцесс):\n"
            f"   - Значение: {kurtosis_value:.4f}\n"
            f"   - Вывод: {kurtosis_interpretation}\n\n"
            f"5. Кластеризация волатильности (ARCH-тест):\n"
            f"   - p-value: {arch_pvalue:.4f}\n"
            f"   - Вывод: {'Присутствует (ARCH-эффект)' if has_arch_effect else 'Отсутствует'}\n"
            f"======================================================"
        )
        self.log.separator()
        self.log.info("ИТОГОВОЕ СТАТИСТИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:\n" + summary_text)
        self.log.separator()

        if mlflow.active_run():
            try:
                summary_path = self.cfg.ARTIFACTS_DIR / "statistical_summary.txt"
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary_text)
                mlflow.log_artifact(str(summary_path), artifact_path="analysis_reports")
                summary_path.unlink()
            except Exception as e:
                self.log.error(f"Не удалось сохранить артефакт с отчетом: {e}")
        
        global_insights = {
            "adf_pvalue": adf_result[1],
            "is_stationary": is_stationary,
            "hurst_exponent": hurst_exponent,
            "lb_pvalue": float(lb_pvalue),
            "has_autocorrelation": has_autocorrelation,
            "kurtosis": float(kurtosis_value),
            "arch_pvalue": arch_pvalue,
            "has_arch_effect": has_arch_effect
        }
        
        self.log.info("Статистическая диагностика завершена.")
        return global_insights

    def fit(self, train_df: pd.DataFrame, global_insights: dict) -> None:
        """
        Обучает параметры для трансформаций (Фаза 2, часть 1).
        Использует ТОЛЬКО тренировочные данные для расчета границ выбросов,
        параметров тренда и т.д.

        Args:
            train_df (pd.DataFrame): Тренировочный DataFrame.
            global_insights (dict): Результаты из фазы анализа.
        """
        self.log.info("Обучение параметров для статистических трансформаций на train-выборке...")
        # Пока заглушка, здесь будет расчет границ для выбросов.
        pass

    def transform(self, df: pd.DataFrame, experiment_cfg: 'ExperimentConfig') -> tuple[pd.DataFrame, bool]:
        """
        Применяет полный, методологически корректный цикл трансформаций.
        1. Логарифмирование.
        2. Промежуточный тест на стационарность.
        3. Условное дифференцирование (если нужно, исключая таргеты).
        4. Финальный контрольный тест (если было дифференцирование).

        Args:
            df (pd.DataFrame): DataFrame для трансформации (train, val или test).
            experiment_cfg (ExperimentConfig): Конфигурация эксперимента.

        Returns:
            Кортеж (pd.DataFrame, bool): (Трансформированный DataFrame, флаг 'было ли дифференцирование').
        """
        self.log.info(f"Применение статистических трансформаций к выборке формы {df.shape}...")
        df_copy = df.copy()
        
        # 1. Логарифмирование OHLCV и ТАРГЕТОВ (применяется всегда)
        targets = experiment_cfg.common_params.get("targets", [])
        target_cols = [f"target_{t}" for t in targets]


        cols_to_log = ['Open', 'High', 'Low', 'Close', 'Volume'] + target_cols

        for col in cols_to_log:        
            if col in df_copy.columns:
                # Добавляем небольшую константу, чтобы избежать log(0) для Volume
                df_copy[col] = np.log(df_copy[col] + 1e-8)

        # 2: Тест на стационарность ПОСЛЕ логарифмирования для ВСЕХ таргетов
        any_target_is_non_stationary = False
        for target_name in targets:
            adf_result = adfuller(df_copy[target_name].dropna())
            p_value = adf_result[1]
            is_stationary = p_value < 0.05
            log_func = self.log.info if is_stationary else self.log.warning
            log_func(f"  Проверка '{target_name}' после логарифмирования (p-value): {p_value:.4f}. Стационарный: {is_stationary}")
            if not is_stationary:
                any_target_is_non_stationary = True
        
        # 3. Применение дифференцирования (если необходимо)
        diff_mode = experiment_cfg.common_params.get("differencing", "false")
        apply_diff = (diff_mode == "true") or (diff_mode == "auto" and any_target_is_non_stationary)

        if apply_diff:
            self.log.warning("Применение первой разности (дифференцирование) для приведения к стационарности.")
            
            # Определяем колонки, которые НЕ нужно дифференцировать (ИСКЛЮЧАЕМ УТЕЧКУ ДАННЫХ)
            cols_to_exclude = [col for col in df_copy.columns if '_sin' in col or '_cos' in col]
            cols_to_exclude.extend([col for col in df_copy.columns if col.startswith('target_')])
            cols_to_exclude.append('year')
            
            # Добавим сюда популярные осцилляторы, которые уже стационарны
            oscillators = ['rsi', 'stoch', 'cci', 'mom', 'roc', 'willr']
            for osc in oscillators:
                cols_to_exclude.extend([col for col in df_copy.columns if col.startswith(osc)])
            
            # Получаем список колонок для дифференцирования
            cols_to_diff = [col for col in df_copy.select_dtypes(include=np.number).columns if col not in set(cols_to_exclude)]

            self.log.info(f"Будут продифференцированы {len(cols_to_diff)} колонок (только признаки).")
            df_copy[cols_to_diff] = df_copy[cols_to_diff].diff()

            # Финальный контрольный тест для КАЖДОГО таргета
            self.log.info("Финальная контрольная проверка стационарности целевых рядов...")
            all_targets_stationary_after_diff = True
            for target_name in targets:
                adf_result = adfuller(df_copy[target_name].dropna())
                p_value = adf_result[1]
                is_stationary = p_value < 0.05
                log_func = self.log.info if is_stationary else self.log.error
                log_func(f"  Проверка '{target_name}' после diff (p-value): {p_value:.4f}. Стационарный: {is_stationary}")
                if not is_stationary:
                    all_targets_stationary_after_diff = False
            
            if all_targets_stationary_after_diff and mlflow.active_run():
                # Логируем p-value только репрезентативного ряда, чтобы не засорять MLflow
                representative_p_value = float(adfuller(df_copy[targets[0]].dropna())[1])
                mlflow.log_metric(f"adf_pvalue_after_diff_{df.shape[0]}", representative_p_value)
        
        # Очистка NaN, появившихся после .diff()
        df_copy.dropna(inplace=True)

        return df_copy, apply_diff
        
    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Рассчитывает показатель Хёрста для временного ряда.

        Args:
            series (pd.Series): Временной ряд (например, цены закрытия).

        Returns:
            float: Значение показателя Хёрста.
        """
        if len(series) < 100:
            return np.nan # Недостаточно данных для надежного расчета

        lags = range(2, 100)
        # Рассчитываем стандартизированный размах
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        # Используем полифит для нахождения наклона в лог-лог масштабе
        poly = np.polyfit(np.log(lags), np.log(np.array(tau) + 1e-8), 1)
        
        return poly[0] * 2.0

  