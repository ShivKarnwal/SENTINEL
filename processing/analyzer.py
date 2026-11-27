
import csv


def calculate_rolling_average(values, window_size=3):

    if window_size <= 0:
        raise ValueError("Window size must be positive")

    if len(values) < window_size:
        return []

    averages = []

    for end_index in range(window_size, len(values) + 1):
        window = values[end_index - window_size:end_index]
        average = sum(window) / window_size
        averages.append(average)

    return averages


def load_temperature_history(csv_path):

    temperatures = []

    with open(csv_path, "r", newline="") as file:
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
    rolling_average=None,
    high_temp_threshold=35,
    deviation_threshold=5
):
    alerts = []

    # Check 1: Fixed temperature threshold
    if current_temperature >= high_temp_threshold:
        alerts.append("High temperature detected")

    # Check 2: Sudden deviation from recent average
    if rolling_average is not None:
        difference = abs(current_temperature - rolling_average)

        if difference >= deviation_threshold:
            alerts.append("Sudden temperature change detected")

    return alerts

