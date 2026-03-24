import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .anomaly_visualization_config import VISUALIZATION_CONFIG
from .trace import log_dataframe, log_event

from .config import PLOTS_DIR

logger = logging.getLogger(__name__)


def visualize_predictions(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    anomalies: pd.DataFrame,
    history_points: int = 1000,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Функция для визуализации исходных данных, предсказаний и аномалий.
    Args:
        df: Исходный DataFrame с историческими данными
        predictions: DataFrame с предсказанными значениями
        anomalies: DataFrame с аномалиями
        history_points: Количество последних точек истории для отображения
    """
    log_event(
        logger,
        "visualize_predictions.start",
        history_points=history_points,
        output_dir=str(output_dir) if output_dir else None,
    )
    log_dataframe(logger, "visualize_predictions.df", df)
    log_dataframe(logger, "visualize_predictions.predictions", predictions)
    log_dataframe(logger, "visualize_predictions.anomalies", anomalies)
    # Подготовка данных для визуализации
    history_data = df.tail(history_points)
    # Создание визуализации
    plt.figure(figsize=VISUALIZATION_CONFIG["main_plot"]["figsize"])
    # Отображение исторических данных
    plt.plot(
        history_data["timestamp"],
        history_data["value"],
        label="Исторические данные",
        color="blue",
        linewidth=2,
    )
    # Отображение предсказанных данных
    plt.plot(
        predictions["timestamp"],
        predictions["predicted"],
        label="Предсказанные значения",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    # Отображение аномалий
    if not anomalies.empty:
        plt.scatter(
            anomalies["timestamp"],
            anomalies["value"],
            color="green",
            s=100,
            zorder=5,
            label="Аномалии",
            marker="x",
            linewidth=2,
        )
    # Настройка графика
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.title("График исторических данных, предсказаний и аномалий")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Создание директории для сохранения графиков
    output_dir = output_dir or Path(VISUALIZATION_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Генерация имени файла с временной меткой
    timestamp = datetime.now().strftime(VISUALIZATION_CONFIG["datetime_format"])
    filename = f"{VISUALIZATION_CONFIG['main_plot']['filename_prefix']}_{timestamp}.png"
    filepath = output_dir / filename
    # Сохранение графика в файл
    plt.savefig(
        filepath,
        dpi=VISUALIZATION_CONFIG["main_plot"]["dpi"],
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Main anomaly visualization saved to {filepath}")
    log_event(logger, "visualize_predictions.done", filepath=str(filepath))
    return filepath


def visualize_combined(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    anomalies: pd.DataFrame,
    history_points: int = 1000,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    График #1: прошлое + будущее + отмеченные аномалии.
    """
    log_event(
        logger,
        "visualize_combined.start",
        history_points=history_points,
        output_dir=str(output_dir) if output_dir else None,
    )
    log_dataframe(logger, "visualize_combined.df", df)
    log_dataframe(logger, "visualize_combined.predictions", predictions)
    log_dataframe(logger, "visualize_combined.anomalies", anomalies)
    output_dir = output_dir or Path(VISUALIZATION_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    history_data = df
    last_actual_ts = history_data["timestamp"].max()
    future_predictions = predictions[predictions["timestamp"] > last_actual_ts].copy()
    forecast_end_ts = None
    if not future_predictions.empty:
        forecast_end_ts = future_predictions["timestamp"].max()

    ax.plot(
        history_data["timestamp"],
        history_data["value"],
        label="Исторические данные",
        color="blue",
        linewidth=2,
    )
    if not future_predictions.empty:
        ax.axvline(
            last_actual_ts,
            color="#6b7280",
            linestyle=":",
            linewidth=1.2,
            label="Начало прогноза",
        )
        ax.axvspan(
            last_actual_ts,
            forecast_end_ts,
            color="#ef4444",
            alpha=0.08,
            label="Прогнозный участок",
        )
        ax.plot(
            future_predictions["timestamp"],
            future_predictions["predicted"],
            label="Предсказанные значения (будущее)",
            color="#ef4444",
            linewidth=2,
            linestyle="--",
        )
    actual_anomalies = anomalies
    if "source" in anomalies.columns:
        actual_anomalies = anomalies[anomalies["source"] == "actual"]
    if not actual_anomalies.empty:
        ax.scatter(
            actual_anomalies["timestamp"],
            actual_anomalies["value"],
            color="green",
            s=80,
            zorder=5,
            label="Аномалии",
            marker="x",
            linewidth=2,
        )
    ax.set_title("Все замеры и аномалии")
    ax.set_xlabel("Время")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    timestamp = datetime.now().strftime(VISUALIZATION_CONFIG["datetime_format"])
    filename = f"combined_anomalies_{timestamp}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=VISUALIZATION_CONFIG["main_plot"]["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined anomaly visualization saved to %s", filepath)
    log_event(logger, "visualize_combined.done", filepath=str(filepath))
    return filepath


def create_anomaly_detail_plot(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    anomaly: pd.Series,
    anomaly_idx: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Создает детальный график для отдельной аномалии с контекстом до и после.
    Args:
        df: DataFrame с историческими данными
        predictions: DataFrame с предсказанными значениями
        anomaly: Series с данными об аномалии
        anomaly_idx: Индекс аномалии для именования файла
    """
    log_event(
        logger,
        "create_anomaly_detail_plot.start",
        anomaly_idx=anomaly_idx,
        output_dir=str(output_dir) if output_dir else None,
    )
    # Получаем временные границы для детального графика
    anomaly_time = anomaly["timestamp"]
    loopback_minutes = VISUALIZATION_CONFIG["anomaly_plot"]["loopback_minutes"]
    start_time = anomaly_time - pd.Timedelta(minutes=loopback_minutes)
    end_time = anomaly_time + pd.Timedelta(minutes=loopback_minutes)
    # Фильтруем данные в пределах временного окна
    mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
    window_data = df[mask].copy()
    # Фильтруем предсказания в пределах временного окна
    pred_mask = (predictions["timestamp"] >= start_time) & (
        predictions["timestamp"] <= end_time
    )
    window_predictions = predictions[pred_mask].copy()
    # Создаем визуализацию
    plt.figure(figsize=VISUALIZATION_CONFIG["anomaly_plot"]["figsize"])
    # Отображение исторических данных
    plt.plot(
        window_data["timestamp"],
        window_data["value"],
        label="Исторические данные",
        color="blue",
        linewidth=2,
    )
    # Отображение предсказанных данных
    plt.plot(
        window_predictions["timestamp"],
        window_predictions["predicted"],
        label="Предсказанные значения",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    # Отображение аномалии
    plt.scatter(
        anomaly["timestamp"],
        anomaly["value"],
        color="green",
        s=100,
        zorder=5,
        label="Аномалия",
        marker="x",
        linewidth=2,
    )
    # Настройка графика
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.title(
        "Детальный график аномалии "
        f"{anomaly_idx + 1} ({anomaly_time.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Создание директории для сохранения графиков
    output_dir = output_dir or Path(VISUALIZATION_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Генерация имени файла с временной меткой
    timestamp = datetime.now().strftime(VISUALIZATION_CONFIG["datetime_format"])
    filename = (
        f"{VISUALIZATION_CONFIG['anomaly_plot']['filename_prefix']}_"
        f"{anomaly_idx + 1}_{timestamp}.png"
    )
    filepath = output_dir / filename
    # Сохранение графика в файл
    plt.savefig(
        filepath,
        dpi=VISUALIZATION_CONFIG["anomaly_plot"]["dpi"],
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Anomaly detail plot {anomaly_idx + 1} saved to {filepath}")
    log_event(logger, "create_anomaly_detail_plot.done", filepath=str(filepath))
    return filepath


def visualize(query: str, api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create visualizations for predictions and anomalies from API response.
    Args:
        query: PromQL query that was analyzed
        api_response: Response data from the API
    Returns:
        List of dictionaries with information about saved plots.
    """
    log_event(logger, "visualize.start", query=query)
    data = api_response or {}
    payload = data.get("data") or {}
    merged_records = payload.get("merged_data") or []
    prediction_records = payload.get("predictions") or []
    anomaly_records = payload.get("anomalies") or []
    logger.info(
        "Creating visualization with merged=%s, predictions=%s, anomalies=%s",
        len(merged_records),
        len(prediction_records),
        len(anomaly_records),
    )
    df = pd.DataFrame(
        merged_records,
        columns=["timestamp", "value", "predicted", "residual", "is_anomaly"],
    )
    predictions = pd.DataFrame(
        prediction_records,
        columns=["timestamp", "predicted"],
    )
    anomalies = pd.DataFrame(
        anomaly_records,
        columns=["timestamp", "value", "predicted", "residual", "is_anomaly", "source"],
    )
    if df.empty or predictions.empty:
        logger.warning(
            "Visualization skipped: merged=%s, predictions=%s",
            len(df),
            len(predictions),
        )
        log_event(logger, "visualize.skipped_empty", merged=len(df), predictions=len(predictions))
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    if not anomalies.empty:
        anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])
    plot_info: List[Dict[str, Any]] = []
    combined_path = visualize_combined(
        df,
        predictions,
        anomalies,
        history_points=1000,
        output_dir=PLOTS_DIR,
    )
    plot_info.append(
        {
            "path": str(combined_path),
            "type": "combined",
            "anomaly_timestamp": None,
            "anomaly_value": None,
        }
    )
    log_event(logger, "visualize.done", plots=len(plot_info))
    return plot_info
