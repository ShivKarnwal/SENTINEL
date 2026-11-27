import random
import csv
from pathlib import Path
from datetime import datetime


DATA_FILE = Path("storage/readings.csv")


def read_simulated_sensors():
    """Generate simulated sensor readings."""

    return {
        "temperature_c": round(random.uniform(20, 40), 2),
        "humidity_pct": round(random.uniform(40, 90), 2),
        "light_raw": random.randint(0, 1023)
    }


def validate_reading(reading):
    """Validate incoming sensor data."""

    temperature = reading["temperature_c"]
    humidity = reading["humidity_pct"]
    light = reading["light_raw"]

    if not -40 <= temperature <= 80:
        raise ValueError("Invalid temperature")

    if not 0 <= humidity <= 100:
        raise ValueError("Invalid humidity")

    if not 0 <= light <= 1023:
        raise ValueError("Invalid light")

    return True


def save_reading(reading):
    """Save readings with a timestamp."""

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    file_exists = DATA_FILE.exists()

    with DATA_FILE.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "temperature_c",
                "humidity_pct",
                "light_raw"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": timestamp,
            **reading
        })

    return timestamp


def main():
    reading = read_simulated_sensors()

    validate_reading(reading)

    timestamp = save_reading(reading)

    print("\n===== SENTINEL =====")
    print("Timestamp:", timestamp)

    for sensor, value in reading.items():
        print(f"{sensor}: {value}")

    print("Reading saved successfully.")


if __name__ == "__main__":
    main()
