"""Environmental analysis helpers used by the SENTINEL dashboard."""

import csv


def calculate_rolling_average(values, window_size=3):
    """Return moving averages for each complete window in values."""
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if len(values) < window_size:
        return []
    averages = []
    for end_index in range(window_size, len(values) + 1):
        window = values[end_index - window_size:end_index]
        averages.append(sum(window) / window_size)
    return averages


def load_temperature_history(csv_path):
    """Load valid temperature values from current or legacy SENTINEL CSV files."""
    temperatures = []
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                value = row.get("temperature")
                if value is None:
                    value = row.get("temperature_c")
                temperatures.append(float(value))
            except (ValueError, TypeError):
                continue
    return temperatures


def detect_temperature_anomalies(
    current_temperature,
    previous_temperature=None,
    high_temp_threshold=35.0,
    deviation_threshold=5.0,
):
    """Detect high temperature and >5 C change in either direction.

    Sudden-change detection compares consecutive valid sensor samples, not a
    rolling average. The threshold is strict: exactly 5.0 C is not an alert.
    """
    current = float(current_temperature)
    alerts = []

    if current > float(high_temp_threshold):
        alerts.append(
            f"High temperature detected: {current:.1f} °C (above {high_temp_threshold:.1f} °C)"
        )

    if previous_temperature is not None:
        previous = float(previous_temperature)
        change = current - previous
        if abs(change) > float(deviation_threshold):
            direction = "rise" if change > 0 else "fall"
            alerts.append(
                f"Sudden temperature {direction}: {previous:.1f} °C to {current:.1f} °C "
                f"({abs(change):.1f} °C change)"
            )

    return alerts
