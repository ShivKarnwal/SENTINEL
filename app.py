import base64
import json
import shutil
import subprocess
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import serial
from serial.tools import list_ports
import streamlit as st

# Optional local AI/webcam dependencies. The environmental dashboard can still
# run if these packages are not installed; the AI panel will explain what's missing.
try:
    import cv2
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models
    from torchvision.models import MobileNet_V3_Small_Weights
    AI_IMPORT_ERROR = None
except Exception as ai_import_exc:
    cv2 = None
    torch = None
    nn = None
    Image = None
    models = None
    MobileNet_V3_Small_Weights = None
    AI_IMPORT_ERROR = str(ai_import_exc)

from processing.analyzer import calculate_rolling_average, detect_temperature_anomalies

BASE = Path(__file__).resolve().parent
BG = BASE / "assets" / "plant_bg.jpg"
CSV = BASE / "storage" / "dashboard_readings.csv"
SETTINGS_FILE = BASE / "storage" / "sentinel_settings.json"
AI_ROOT = BASE / "processing" / "SENTINEL_AI"
AI_MODEL_PATH = AI_ROOT / "models" / "sentinel_mobilenetv3_small_finetuned_best.pth"
MAX_CSV_ROWS = 10_000
RETENTION_CHECK_INTERVAL = 100
ARDUINO_BAUD_RATE = 9600
ARDUINO_STALE_SECONDS = 5.5
ARDUINO_FQBN = "arduino:avr:uno"
ARDUINO_SKETCH_DIR = BASE / "acquisition" / "arduino_firmware" / "SENTINEL_Sensors"
ARDUINO_SKETCH_PATH = ARDUINO_SKETCH_DIR / "SENTINEL_Sensors.ino"
ZERO_READING = {"temperature": 0.0, "humidity": 0.0, "light": 0}


def load_saved_app_settings():
    """Load persisted SENTINEL settings without failing on a missing file."""
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {}


def save_app_settings(updates):
    """Update settings atomically while preserving unrelated settings."""
    data = load_saved_app_settings()
    data.update(updates)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_FILE.with_suffix(SETTINGS_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)


def send_ntfy_notification(topic, title, message, priority="high", tags="warning"):
    """Send one push notification through ntfy.sh using its HTTP publish API."""
    topic = str(topic or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", topic):
        return False, "Enter a topic containing 8–64 letters, numbers, hyphens or underscores."

    url = "https://ntfy.sh/" + urllib.parse.quote(topic, safe="")
    request = urllib.request.Request(
        url,
        data=str(message).encode("utf-8"),
        headers={
            "Title": str(title),
            "Priority": str(priority),
            "Tags": str(tags),
            "Content-Type": "text/plain; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            if 200 <= response.status < 300:
                return True, ""
            return False, f"ntfy returned HTTP {response.status}."
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)

st.set_page_config(page_title="SENTINEL | Plant Environment", page_icon="🌱", layout="wide")

def background():
    if BG.exists():
        encoded = base64.b64encode(BG.read_bytes()).decode("ascii")
        return (
            "background-image: linear-gradient(rgba(8,15,14,.79),rgba(8,15,14,.88)),"
            f'url("data:image/jpeg;base64,{encoded}");'
        )
    return "background:#101817;"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700;800&family=Roboto:wght@400;500;600;700&display=swap');
html,body,[class*="css"] {{font-family:'Roboto',sans-serif;}}
.stApp {{{background()} background-size:cover;background-position:center;background-attachment:fixed;color:#eee9df;}}
[data-testid="stHeader"] {{background:transparent;}}
.block-container {{max-width:1600px;padding-top:1.7rem;padding-bottom:2rem;}}
h1,h2,h3,h4 {{font-family:'Inter',sans-serif!important;font-weight:700!important;letter-spacing:-.45px;color:#f0eadd!important;}}
[data-testid="stSidebar"] {{min-width:255px;max-width:255px;background:rgba(8,16,15,.97);border-right:1px solid rgba(255,255,255,.10);}}
[data-testid="stSidebar"]>div:first-child {{padding:1.8rem 1rem 1rem;}}
.brand {{font-family:'Inter',sans-serif;font-size:25px;font-weight:800;letter-spacing:1.2px;color:#f0eadf;white-space:nowrap;}}
.brand-subtitle {{color:#8f9b94;font-size:13px;line-height:1.55;margin:7px 0 0 39px;}}
.sidebar-rule {{border:0;border-top:1px solid rgba(255,255,255,.15);margin:25px 0;}}
[data-testid="stSidebar"] [data-testid="stRadio"]>label {{display:none;}}
[data-testid="stSidebar"] [data-testid="stRadio"]>div {{gap:7px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{padding:11px 12px;border-radius:11px;background:transparent;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{background:rgba(126,171,137,.13);}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{background:rgba(108,153,119,.28);outline:1px solid rgba(132,184,143,.18);}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{color:#e2e4dc;font-size:14px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label>div:first-child {{display:none;}}
.sidebar-status {{background:rgba(27,39,36,.88);border:1px solid rgba(210,225,210,.14);border-radius:14px;padding:15px;margin:8px 0 20px;}}
.green-text,.status-line {{color:#83c99a;}}
.muted {{color:#929c96;font-size:12px;}}
.panel {{background:rgba(20,29,28,.84);border:1px solid rgba(205,220,205,.18);border-radius:16px;padding:19px;backdrop-filter:blur(13px);box-shadow:0 8px 28px rgba(0,0,0,.15);margin-bottom:12px;}}
.metric-label {{color:#c4c6bd;font-size:14px;margin-top:5px;}}
.metric-value {{font-family:'Inter',sans-serif;font-size:clamp(27px,2.5vw,38px);font-weight:600;color:#f3eee3;margin:8px 0 4px;}}
.metric-icon {{font-size:25px;margin-bottom:7px;}}
.section-title {{font-family:'Inter',sans-serif;font-size:18px;font-weight:700;color:#eee9df;margin-bottom:12px;}}
.status-line {{font-size:14px;font-weight:600;}}
.plant-illustration {{height:205px;display:flex;align-items:center;justify-content:center;margin:3px auto 8px;}}
.plant-scene {{width:174px;height:174px;position:relative;border:2px solid rgba(132,184,143,.76);border-radius:50%;background:radial-gradient(circle,rgba(110,155,117,.13),transparent 68%);box-shadow:0 0 24px rgba(116,171,126,.10);}}
.plant-stem {{position:absolute;left:84px;top:43px;width:5px;height:83px;border-radius:5px;background:#79ad7d;}}
.plant-leaf {{position:absolute;width:48px;height:27px;border-radius:100% 0 100% 0;background:#72a77a;}}
.leaf-left {{left:39px;top:61px;transform:rotate(12deg);}}
.leaf-right {{left:88px;top:38px;width:53px;height:30px;transform:rotate(-28deg) scaleX(-1);background:#8cbd89;}}
.leaf-small {{left:48px;top:91px;width:39px;height:23px;transform:rotate(8deg);background:#547f5e;}}
.plant-pot {{position:absolute;left:49px;bottom:17px;width:76px;height:49px;background:linear-gradient(135deg,#778d7d,#40594d);clip-path:polygon(0 0,100% 0,86% 100%,14% 100%);}}
.plant-pot-rim {{position:absolute;left:45px;bottom:61px;width:84px;height:10px;border-radius:4px;background:#91a695;}}
.comfort-status {{color:#8bce9f;font-family:'Inter',sans-serif;font-size:17px;font-weight:700;text-align:center;}}
.comfort-description {{text-align:center;color:#c2c5bc;font-size:12px;line-height:1.55;margin-top:8px;}}
div[data-testid="stMetric"] {{background:rgba(255,255,255,.035);padding:12px;border-radius:12px;}}
div[data-testid="stMetricValue"] {{color:#f1ebdf;}}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input {{background-color:rgba(18,25,24,.9);color:#eee9df;}}
</style>
""", unsafe_allow_html=True)

# Alert visuals are injected only while an environmental alert is active.
ALERT_STYLE = """
<style>
.stApp { background: radial-gradient(circle at center, #7f1010 0%, #390606 65%, #170202 100%) !important; }
[data-testid="stHeader"] { background: rgba(70,0,0,.35) !important; }
.sentinel-alert-overlay {
 position: fixed; inset: 0; z-index: 999999; pointer-events: none;
 border: 10px solid #ff2424; box-shadow: inset 0 0 100px 25px rgba(255,0,0,.55);
 animation: sentinel-alert-pulse .75s infinite alternate;
}
.sentinel-alert-banner {
 position: fixed; z-index: 1000000; top: 18px; left: 50%; transform: translateX(-50%);
 width: min(760px, 92vw); padding: 18px 24px; border: 2px solid #fff;
 border-radius: 14px; background: rgba(145,0,0,.97); color: #fff;
 text-align: center; box-shadow: 0 0 35px rgba(255,0,0,.85);
 font-family: Arial, sans-serif; animation: sentinel-alert-pulse .75s infinite alternate;
}
.sentinel-alert-banner h2 { color:#fff !important; margin:0 0 6px 0; font-size: clamp(22px,3vw,36px); }
.sentinel-alert-banner p { color:#fff; margin:0; font-weight:700; }
@keyframes sentinel-alert-pulse { from { opacity:.78; } to { opacity:1; } }
</style>
"""

# ---------------------------- SENTINEL AI / Webcam ----------------------------
@st.cache_resource(show_spinner=False)
def load_sentinel_model():
    """Load the fine-tuned MobileNetV3-Small checkpoint once per app process."""
    if AI_IMPORT_ERROR:
        raise RuntimeError(
            "AI dependencies are unavailable. Install torch, torchvision, "
            "Pillow and opencv-python in the dashboard environment. "
            f"Import detail: {AI_IMPORT_ERROR}"
        )

    if not AI_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {AI_MODEL_PATH}"
        )

    checkpoint = torch.load(
        AI_MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    return model, transform, class_names, checkpoint


@st.cache_resource(show_spinner=False)
def get_webcam():
    """Open the Ubuntu/VirtualBox webcam once and reuse it across refreshes."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed in the Python environment running Streamlit."
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Ubuntu could not open camera index 0. Check VirtualBox webcam "
            "USB passthrough and close other apps using the camera."
        )

    # Request a modest frame size to reduce CPU and USB overhead.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def release_webcam():
    """Release the camera when the user turns webcam monitoring off."""
    camera = st.session_state.get("_sentinel_webcam_object")
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
        st.session_state.pop("_sentinel_webcam_object", None)
    try:
        get_webcam.clear()
    except Exception:
        pass


def predict_plant_frame(frame):
    """Return top-3 PlantVillage class predictions for one webcam frame."""
    model, transform, class_names, checkpoint = load_sentinel_model()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    input_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        scores, indices = torch.topk(
            probabilities, k=min(3, len(class_names))
        )

    results = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        results.append((class_names[index], float(score)))

    return results, checkpoint


def render_webcam_ai_panel():
    """Render the webcam box and model predictions inside the Overview page."""
    st.markdown("---")
    st.subheader("SENTINEL AI · Live Plant Analysis")
    st.caption(
        "The webcam frame is classified by the original fine-tuned 38-class "
        "MobileNetV3-Small model. Low-confidence frames are flagged as "
        "possibly non-plant / uncertain, not definitively plant-free."
    )

    # Start webcam analysis automatically on a fresh dashboard session.
    # Users can still turn it off with the checkbox.
    if "sentinel_webcam_enabled" not in st.session_state:
        st.session_state["sentinel_webcam_enabled"] = True

    enabled = st.checkbox(
        "Enable webcam and AI analysis",
        key="sentinel_webcam_enabled",
        help="Webcam and AI analysis start automatically. Uncheck to turn them off."
    )

    camera_col, result_col = st.columns([1.45, 1], gap="large")

    with camera_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live Webcam</div>', unsafe_allow_html=True)

        if not enabled:
            release_webcam()
            st.session_state.pop("_sentinel_current_frame", None)
            st.info("Webcam is off. Enable it above to display the live camera feed.")
            st.markdown(
                '<div class="muted">Camera: Ubuntu / VirtualBox · Index 0</div>',
                unsafe_allow_html=True
            )
        else:
            try:
                camera = get_webcam()
                st.session_state["_sentinel_webcam_object"] = camera
                success, frame = camera.read()

                if not success or frame is None:
                    st.session_state.pop("_sentinel_current_frame", None)
                    st.warning("Camera opened, but no frame was received. Check the webcam connection.")
                else:
                    st.session_state["_sentinel_current_frame"] = frame.copy()
                    st.image(
                        frame,
                        channels="BGR",
                        use_container_width=True,
                        caption="Current webcam frame"
                    )
                    st.caption("The camera refreshes with the dashboard update interval.")
            except Exception as exc:
                st.session_state.pop("_sentinel_current_frame", None)
                st.error(f"Webcam error: {exc}")
                release_webcam()

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Prediction</div>', unsafe_allow_html=True)

        if not enabled:
            st.info("Enable the webcam to start AI analysis.")
        elif "_sentinel_current_frame" not in st.session_state:
            st.info("Waiting for a usable webcam frame.")
        else:
            try:
                frame = st.session_state["_sentinel_current_frame"]
                with st.spinner("Analyzing plant image..."):
                    predictions, checkpoint = predict_plant_frame(frame)

                    if predictions:
                        top_name, top_score = predictions[0]
                        display_name = top_name.replace("___", " | ").replace("_", " ")

                        confidence_threshold = st.slider(
                            "Plant-match confidence threshold",
                            min_value=0.30,
                            max_value=0.95,
                            value=0.70,
                            step=0.05,
                            help=(
                                "If the strongest plant-class score is below this value, "
                                "SENTINEL marks the frame as uncertain / possibly non-plant. "
                                "This is a demo threshold, not a validated detector."
                            ),
                            key="sentinel_plant_confidence_threshold"
                        )

                        st.metric("Highest plant-class score", f"{top_score * 100:.1f}%")

                        if top_score < confidence_threshold:
                            st.warning("POSSIBLE NO PLANT / UNCERTAIN IMAGE")
                            st.markdown(
                                "**No plant class matched confidently.** The camera may be "
                                "seeing a person, an empty scene, an unfamiliar plant, or a "
                                "difficult image (for example, blur or poor lighting)."
                            )
                            st.caption(
                                "This is a low-confidence warning, NOT proof that no plant is "
                                "present. A real plant can also receive a low score."
                            )
                        else:
                            st.markdown(
                                f'<div class="muted">MOST LIKELY PLANT CLASS</div>'
                                f'<div style="font-size:20px;font-weight:700;color:#f0eadd;margin:8px 0">'
                                f'{display_name}</div>',
                                unsafe_allow_html=True
                            )
                            st.success("A plant class matched above the selected threshold.")

                        with st.expander("View the model's closest plant-class guesses"):
                            for rank, (name, score) in enumerate(predictions, start=1):
                                readable = name.replace("___", " | ").replace("_", " ")
                                st.write(f"{rank}. {readable} — {score * 100:.1f}%")

                        st.caption(
                            f"Checkpoint: original fine-tuned 38-class model · "
                            f"Epoch {checkpoint.get('epoch', 'unknown')} · "
                            "The experimental 39-class checkpoint is not used."
                        )
                    else:
                        st.info("The model did not return a prediction.")
            except Exception as exc:
                st.error(f"AI analysis error: {exc}")
                st.caption(
                    "Check that the checkpoint exists and that torch, torchvision, "
                    "Pillow and OpenCV are installed in the environment running Streamlit."
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "Important: the displayed score is a softmax score, not a calibrated "
        "probability. The low-confidence warning is only a best-effort unknown-scene "
        "heuristic; it can flag real plants as uncertain and cannot prove that a "
        "plant is absent. PlantVillage test accuracy does not guarantee webcam accuracy."
    )

def parse_sensor_line(line):
    """Convert one CSV serial line into validated sensor values."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected temperature,humidity,light; received {line!r}")
    temperature, humidity, light = map(float, parts)
    if not -40 <= temperature <= 80:
        raise ValueError(f"Temperature outside accepted range: {temperature}")
    if not 0 <= humidity <= 100:
        raise ValueError(f"Humidity outside accepted range: {humidity}")
    if not 0 <= light <= 1023:
        raise ValueError(f"Light ADC value outside Uno range: {light}")
    return {
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "light": int(light),
    }


def classify_light_level(adc_value):
    """Classify the user's LDR readings; return 0 when no valid sample exists."""
    value = int(adc_value)
    if value <= 0:
        return "0"
    # Calibrated to the observed module values: bright ~75, moderate ~350,
    # dark ~750. These are practical starting boundaries, not lux values.
    if value <= 200:
        return "BRIGHT"
    if value <= 550:
        return "MODERATE"
    return "DARK"


SENTINEL_ARDUINO_SKETCH = r"""#include <DHT.h>

// SENTINEL firmware for Arduino Uno
// DHT22 DATA -> D2; LM35 VOUT -> A0; LDR module AO -> A1
#define DHT_PIN 2
#define DHT_TYPE DHT22
#define LM35_PIN A0
#define LDR_PIN A1

DHT dht(DHT_PIN, DHT_TYPE);
const unsigned long SENSOR_INTERVAL_MS = 2000;
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(LDR_PIN, INPUT);
  delay(1000);
  Serial.println("SENTINEL_READY");
}

void loop() {
  const unsigned long now = millis();
  if (now - previousMillis < SENSOR_INTERVAL_MS) return;
  previousMillis = now;

  const float temperature = dht.readTemperature();
  const float humidity = dht.readHumidity();
  const int lm35Raw = analogRead(LM35_PIN); // retained for future diagnostics
  const int lightADC = analogRead(LDR_PIN);
  (void)lm35Raw;

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("DHT22_ERROR");
    return;
  }

  // CSV protocol consumed by app.py: temperature,humidity,lightADC
  Serial.print(temperature, 1);
  Serial.print(',');
  Serial.print(humidity, 1);
  Serial.print(',');
  Serial.println(lightADC);
}
"""


def find_arduino_cli():
    """Find Arduino CLI on PATH or in the project's tools directory."""
    executable = shutil.which("arduino-cli") or shutil.which("arduino-cli.exe")
    if executable:
        return executable
    local_copy = BASE / "tools" / "arduino-cli.exe"
    return str(local_copy) if local_copy.exists() else None


def run_arduino_cli(args, timeout=240):
    """Run Arduino CLI and return its completed process; raise useful errors."""
    cli = find_arduino_cli()
    if not cli:
        raise RuntimeError(
            "Arduino CLI is not installed or not on PATH. Install it once with "
            "`winget install -e --id ArduinoSA.CLI`, then close and reopen PowerShell."
        )
    try:
        result = subprocess.run(
            [cli, *args], capture_output=True, text=True, timeout=timeout,
            cwd=str(BASE), check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Arduino CLI timed out while running: {' '.join(args)}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "No details returned by Arduino CLI").strip()
        raise RuntimeError(detail[-3000:])
    return result


def ensure_arduino_firmware_uploaded(port):
    """Create, compile and upload SENTINEL firmware to the connected Uno."""
    ARDUINO_SKETCH_DIR.mkdir(parents=True, exist_ok=True)
    if not ARDUINO_SKETCH_PATH.exists() or ARDUINO_SKETCH_PATH.read_text(encoding="utf-8") != SENTINEL_ARDUINO_SKETCH:
        ARDUINO_SKETCH_PATH.write_text(SENTINEL_ARDUINO_SKETCH, encoding="utf-8")

    # Install the Uno board core and sensor library automatically when missing.
    core_list = run_arduino_cli(["core", "list"], timeout=90).stdout
    if "arduino:avr" not in core_list:
        run_arduino_cli(["core", "update-index"], timeout=180)
        run_arduino_cli(["core", "install", "arduino:avr"], timeout=300)

    run_arduino_cli(["lib", "update-index"], timeout=180)
    library_list = run_arduino_cli(["lib", "list"], timeout=120).stdout.lower()
    if "dht sensor library" not in library_list:
        run_arduino_cli(["lib", "install", "DHT sensor library"], timeout=240)
    library_list = run_arduino_cli(["lib", "list"], timeout=120).stdout.lower()
    if "adafruit unified sensor" not in library_list:
        run_arduino_cli(["lib", "install", "Adafruit Unified Sensor"], timeout=240)

    run_arduino_cli(
        ["compile", "--fqbn", ARDUINO_FQBN, str(ARDUINO_SKETCH_DIR)], timeout=300
    )
    run_arduino_cli(
        ["upload", "-p", port, "--fqbn", ARDUINO_FQBN, str(ARDUINO_SKETCH_DIR)], timeout=180
    )
    # Upload resets the Uno; give its bootloader and DHT22 time to start.
    time.sleep(2.5)


def get_available_serial_ports():
    """Return currently available serial port devices, such as COM3 or COM5."""
    try:
        return sorted(list_ports.comports(), key=lambda item: item.device)
    except Exception:
        return []


def resolve_arduino_port(port_choice):
    """Resolve the selected port only while that USB serial device is present."""
    ports = get_available_serial_ports()
    if port_choice and port_choice != "Auto-detect":
        return port_choice if any(item.device == port_choice for item in ports) else None

    if not ports:
        return None

    # Prefer common Arduino / USB-to-serial descriptions, then use the only or
    # first available serial device as a fallback.
    keywords = ("arduino", "ch340", "usb serial", "usb-serial", "ftdi", "cp210")
    for item in ports:
        description = " ".join(
            str(value or "") for value in
            (item.description, item.manufacturer, item.product)
        ).lower()
        if any(keyword in description for keyword in keywords):
            return item.device
    return ports[0].device


@st.cache_resource(show_spinner=False)
def get_serial_connection(port, baud_rate=ARDUINO_BAUD_RATE):
    """Keep one serial connection open instead of resetting the Uno every refresh."""
    return serial.Serial(port, baud_rate, timeout=0.25)


def close_arduino_connection(clear_cache=False):
    """Close this session's active serial handle and optionally clear its cache."""
    device = st.session_state.get("_arduino_device")
    if device is not None:
        try:
            if device.is_open:
                device.close()
        except Exception:
            pass
    st.session_state.pop("_arduino_device", None)
    st.session_state.pop("_active_arduino_port", None)
    if clear_cache:
        try:
            get_serial_connection.clear()
        except Exception:
            pass


def read_available_arduino_readings(port):
    """Read every complete CSV sample currently queued by the Arduino Uno."""
    previous_port = st.session_state.get("_active_arduino_port")
    if previous_port != port:
        close_arduino_connection(clear_cache=True)
        st.session_state["_active_arduino_port"] = port

    device = get_serial_connection(port)
    st.session_state["_arduino_device"] = device
    if not device.is_open:
        raise serial.SerialException(f"Serial port {port} is not open")

    lines = []
    # Do not reset the input buffer: that would discard samples between refreshes.
    first_line = device.readline().decode("utf-8", errors="ignore").strip()
    if first_line:
        lines.append(first_line)

    while device.in_waiting:
        extra = device.readline().decode("utf-8", errors="ignore").strip()
        if not extra:
            break
        lines.append(extra)

    readings = []
    for line in lines:
        # Keep the most recent raw line visible for troubleshooting when the
        # Arduino sends an error/status message instead of the expected CSV.
        st.session_state["last_serial_message"] = line
        try:
            readings.append(parse_sensor_line(line))
        except (ValueError, TypeError):
            # Ignore startup text, partial lines, and malformed serial messages.
            continue
    return readings

def trim_csv_history(max_rows=MAX_CSV_ROWS):
    """Keep the newest max_rows records, preserving the CSV header."""
    if not CSV.exists():
        return 0

    try:
        history = pd.read_csv(CSV)
    except (OSError, ValueError, pd.errors.ParserError):
        # Never overwrite a file that cannot be read safely.
        return 0

    if len(history) <= max_rows:
        return 0

    # Keep one untouched copy from before the first automatic cleanup.
    backup = CSV.with_name(CSV.name + ".pre_retention_backup")
    if not backup.exists():
        shutil.copy2(CSV, backup)

    removed = len(history) - max_rows
    temporary = CSV.with_name(CSV.name + ".tmp")
    history.tail(max_rows).to_csv(temporary, index=False)
    temporary.replace(CSV)  # Atomic replacement on the same filesystem.
    return removed


def save_readings(readings):
    """Append a batch of samples; retention is checked periodically."""
    if not readings:
        return
    CSV.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    rows = []
    for reading in readings:
        rows.append({"timestamp": now.isoformat(timespec="milliseconds"), **reading})
    pd.DataFrame(rows).to_csv(
        CSV, mode="a", header=not CSV.exists(), index=False
    )

    # Avoid rereading the whole CSV after every sensor sample.
    st.session_state.retention_check_counter += len(readings)
    if st.session_state.retention_check_counter >= RETENTION_CHECK_INTERVAL:
        removed = trim_csv_history()
        st.session_state.retention_check_counter = 0
        if removed:
            st.session_state.last_retention_removed = removed


def load_history():
    """Load CSV history and normalize older column names to app names.

    Earlier SENTINEL CSV files used temperature_c, humidity_pct, and
    light_raw. Current readings use temperature, humidity, and light.
    Normalize both formats so old history remains visible in every chart.
    """
    columns = ["timestamp", "temperature", "humidity", "light"]
    if not CSV.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(CSV)
        df.columns = df.columns.str.strip()

        aliases = {
            "temperature_c": "temperature",
            "humidity_pct": "humidity",
            "light_raw": "light",
        }
        for old_name, current_name in aliases.items():
            if current_name not in df.columns and old_name in df.columns:
                df = df.rename(columns={old_name: current_name})

        # Ensure a stable schema even if an older CSV omitted a sensor field.
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for column in ("temperature", "humidity", "light"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    except (OSError, ValueError, pd.errors.ParserError, KeyError):
        return pd.DataFrame(columns=columns)


# ---------------------------- Session state ----------------------------
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = ZERO_READING.copy()
if "last_reading_time" not in st.session_state:
    st.session_state.last_reading_time = None
if "serial_error" not in st.session_state:
    st.session_state.serial_error = None
if "last_serial_message" not in st.session_state:
    st.session_state.last_serial_message = ""
if "previous_valid_temperature" not in st.session_state:
    st.session_state.previous_valid_temperature = None
if "high_temperature_active" not in st.session_state:
    st.session_state.high_temperature_active = False
if "alert_sound_token" not in st.session_state:
    st.session_state.alert_sound_token = 0
if "sudden_change_alert_until" not in st.session_state:
    st.session_state.sudden_change_alert_until = 0.0
if "sudden_change_alert_text" not in st.session_state:
    st.session_state.sudden_change_alert_text = ""
if "sudden_change_active" not in st.session_state:
    st.session_state.sudden_change_active = False
if "arduino_firmware_status" not in st.session_state:
    st.session_state.arduino_firmware_status = "Firmware upload has not started."
if "arduino_firmware_uploaded_port" not in st.session_state:
    st.session_state.arduino_firmware_uploaded_port = None
if "arduino_firmware_attempted_port" not in st.session_state:
    st.session_state.arduino_firmware_attempted_port = None
def load_saved_temperature_threshold():
    """Load the user's threshold from disk so page/session reloads cannot reset it."""
    try:
        if SETTINGS_FILE.exists():
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            value = float(saved.get("high_temperature_threshold", 35.0))
            if 0.0 <= value <= 100.0:
                return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return 35.0


def save_temperature_threshold():
    """Persist a changed threshold immediately when the Settings input changes."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"high_temperature_threshold": float(st.session_state.high_temperature_threshold)}
        SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        st.session_state.temperature_threshold_save_error = str(exc)
    else:
        st.session_state.temperature_threshold_save_error = ""


if "high_temperature_threshold" not in st.session_state:
    st.session_state.high_temperature_threshold = load_saved_temperature_threshold()
if "ntfy_topic" not in st.session_state:
    st.session_state.ntfy_topic = str(load_saved_app_settings().get("ntfy_topic", ""))
if "ntfy_enabled" not in st.session_state:
    st.session_state.ntfy_enabled = bool(load_saved_app_settings().get("ntfy_enabled", False))
if "ntfy_previous_alerts" not in st.session_state:
    st.session_state.ntfy_previous_alerts = {}
if "ntfy_last_error" not in st.session_state:
    st.session_state.ntfy_last_error = ""
if "ntfy_last_sent" not in st.session_state:
    st.session_state.ntfy_last_sent = "No notifications sent in this session yet."
if "force_firmware_upload" not in st.session_state:
    st.session_state.force_firmware_upload = False
if "retention_check_counter" not in st.session_state:
    st.session_state.retention_check_counter = 0
if "last_retention_removed" not in st.session_state:
    st.session_state.last_retention_removed = 0
if "retention_startup_checked" not in st.session_state:
    # Enforce the limit once when this browser session starts, too.
    st.session_state.last_retention_removed = trim_csv_history()
    st.session_state.retention_startup_checked = True


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.markdown('<div class="brand">🌱 SENTINEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Plant Environment<br>Monitoring System</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)

    pages = {
        "⌂　 Live Sensors": "Overview",
        "⌁　 History": "History",
        "♧　 Alerts": "Alerts",
        "⚙　 Settings": "Settings",
    }
    selected = st.radio("Navigation", list(pages), label_visibility="collapsed", key="navigation")
    page = pages[selected]

    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-status"><span class="green-text">● System Online</span>'
        '<div class="muted" style="margin-top:6px">Local application running</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="muted">PLANT ZONE</div>', unsafe_allow_html=True)
    plant_zones = {
        "Greenhouse A": "GH-A-001",
        "Greenhouse B": "GH-B-002",
        "Greenhouse C": "GH-C-003",
        "Greenhouse D": "GH-D-004",
        "Greenhouse E": "GH-E-005",
    }
    selected_zone = st.selectbox(
        "Plant zone",
        list(plant_zones.keys()),
        label_visibility="collapsed",
        key="selected_plant_zone",
    )
    st.caption(f"ID: {plant_zones[selected_zone]}")
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)

    detected_ports = get_available_serial_ports()
    port_options = ["Auto-detect"] + [item.device for item in detected_ports]
    port_choice = st.selectbox(
        "Arduino Uno Port",
        port_options,
        index=0,
        key="arduino_port_choice",
        help="Auto-detect scans for the Arduino Uno USB serial connection."
    )
    if st.button("Retry Arduino firmware upload", use_container_width=True):
        st.session_state.force_firmware_upload = True
        st.session_state.arduino_firmware_uploaded_port = None
        st.session_state.arduino_firmware_attempted_port = None
    refresh_rate = st.selectbox("Dashboard update rate", ["1 second", "2 seconds"], index=0)
    alarm_sound_enabled = st.checkbox(
        "Enable alert sound", value=True, key="alarm_sound_enabled",
        help="The browser may require one interaction with this page before it permits alarm audio."
    )
    refresh_seconds = 1 if refresh_rate == "1 second" else 2
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("SENTINEL Local Prototype")


# ---------------------------- Static page heading ----------------------------
st.title("Plant Environment Overview")
st.caption("Live values update independently; the dashboard layout stays in place.")


# ---------------------------- Partial live update ----------------------------
@st.fragment(run_every=f"{refresh_seconds}s")
def live_panel():
    """Refresh only live sensor content while keeping the rest of the page still."""
    new_readings = []
    error = None
    port = resolve_arduino_port(port_choice)

    if port is None:
        close_arduino_connection(clear_cache=True)
        st.session_state.latest_reading = ZERO_READING.copy()
        st.session_state.last_reading_time = None
        st.session_state.previous_valid_temperature = None
        st.session_state.high_temperature_active = False
        st.session_state.sudden_change_alert_until = 0.0
        st.session_state.sudden_change_active = False
        st.session_state.serial_error = "Arduino Uno not detected"
        st.session_state.arduino_firmware_uploaded_port = None
        st.session_state.arduino_firmware_attempted_port = None
        st.session_state.arduino_firmware_status = "Waiting for Arduino Uno USB connection."
    else:
        try:
            # Automatically compile and flash firmware once per detected port.
            # A manual retry button is provided if the first upload fails.
            if (
                st.session_state.arduino_firmware_uploaded_port != port
                and st.session_state.arduino_firmware_attempted_port != port
            ) or st.session_state.force_firmware_upload:
                close_arduino_connection(clear_cache=True)
                st.session_state.arduino_firmware_attempted_port = port
                st.session_state.arduino_firmware_status = f"Preparing firmware for {port}..."
                ensure_arduino_firmware_uploaded(port)
                st.session_state.arduino_firmware_uploaded_port = port
                st.session_state.force_firmware_upload = False
                st.session_state.arduino_firmware_status = f"SENTINEL firmware uploaded successfully to {port}."
            new_readings = read_available_arduino_readings(port)
            st.session_state.serial_error = None
        except (serial.SerialException, OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
            st.session_state.serial_error = error
            st.session_state.arduino_firmware_status = f"Firmware / connection setup failed: {error}"
            close_arduino_connection(clear_cache=True)
            st.session_state.latest_reading = ZERO_READING.copy()
            st.session_state.last_reading_time = None
            st.session_state.previous_valid_temperature = None
            st.session_state.high_temperature_active = False
            st.session_state.sudden_change_alert_until = 0.0
            st.session_state.sudden_change_active = False

    if new_readings:
        now = datetime.now()
        for item in new_readings:
            current_temp = float(item["temperature"])
            previous_temp = st.session_state.previous_valid_temperature
            anomaly_messages = detect_temperature_anomalies(
                current_temperature=current_temp,
                previous_temperature=previous_temp,
                high_temp_threshold=float(st.session_state.high_temperature_threshold),
                deviation_threshold=5.0,
            )
            sudden_messages = [m for m in anomaly_messages if m.startswith("Sudden")]
            if sudden_messages:
                st.session_state.sudden_change_alert_text = sudden_messages[-1]
                st.session_state.sudden_change_active = True
                st.session_state.alert_sound_token += 1
            elif (
                st.session_state.get("sudden_change_active", False)
                and previous_temp is not None
                and abs(current_temp - float(previous_temp)) <= 5.0
                and current_temp <= float(st.session_state.high_temperature_threshold)
            ):
                # Keep the sudden-change alarm latched until a subsequent
                # valid sample is stable and the temperature is back to normal.
                st.session_state.sudden_change_active = False
            # High temperature is a sustained condition. Sound only when it first activates.
            was_high = bool(st.session_state.get("high_temperature_active", False))
            is_high = current_temp > float(st.session_state.high_temperature_threshold)
            if is_high and not was_high:
                st.session_state.alert_sound_token += 1
            st.session_state.high_temperature_active = is_high
            st.session_state.previous_valid_temperature = current_temp
            record = {"timestamp": now, **item}
            st.session_state.session_history.append(record)
        st.session_state.session_history = st.session_state.session_history[-1000:]
        st.session_state.latest_reading = new_readings[-1]
        st.session_state.last_reading_time = now
        save_readings(new_readings)

    # If the Arduino stops sending data, do not leave stale values on screen.
    last_update = st.session_state.last_reading_time
    if last_update is not None:
        age = (datetime.now() - last_update).total_seconds()
        if age > ARDUINO_STALE_SECONDS:
            st.session_state.latest_reading = ZERO_READING.copy()
            st.session_state.last_reading_time = None
            st.session_state.previous_valid_temperature = None
            st.session_state.high_temperature_active = False
            st.session_state.sudden_change_alert_until = 0.0
            st.session_state.sudden_change_active = False

    reading = st.session_state.latest_reading or ZERO_READING.copy()
    connected = port is not None and st.session_state.serial_error is None

    firmware_status = st.session_state.get("arduino_firmware_status", "")
    if firmware_status:
        if "uploaded successfully" in firmware_status:
            st.success(firmware_status)
        elif "failed" in firmware_status.lower():
            st.error(firmware_status)
        else:
            st.info(firmware_status)

    if error:
        st.error(f"Arduino connection problem: {error}. Sensor values have been reset to zero.")
    elif port is None:
        st.warning("Arduino Uno not detected. Temperature, humidity and light are set to zero.")
    elif st.session_state.last_reading_time is None:
        st.info(
            f"Arduino port {port} selected. Waiting for valid sensor data at "
            f"{ARDUINO_BAUD_RATE} baud; readings remain zero until data arrives."
        )
        last_message = st.session_state.get("last_serial_message", "")
        if last_message:
            st.caption(f"Last Arduino serial message (not accepted as sensor data): {last_message}")
    elif not new_readings:
        st.caption(f"Arduino connected on {port} · waiting for next sensor sample")

    if connected and st.session_state.last_reading_time is not None:
        updated = st.session_state.last_reading_time
        st.markdown(
            '<div class="status-line">● LIVE ARDUINO CONNECTION</div>'
            f'<div class="muted">Port: {port} · Last updated: {updated.strftime("%I:%M:%S %p")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="muted">● ARDUINO OFFLINE · Current sensor values: 0</div>',
            unsafe_allow_html=True,
        )

    # Build the active alert list from the latest valid live reading and the
    # latched sudden-change event. Sudden rise AND sudden fall are treated equally.
    active_alerts = []
    if connected and st.session_state.last_reading_time is not None:
        current_temp = float(reading["temperature"])
        if current_temp > float(st.session_state.high_temperature_threshold):
            active_alerts.append(
                f"HIGH TEMPERATURE: {current_temp:.1f} °C "
                f"(limit: {float(st.session_state.high_temperature_threshold):g} °C)"
            )
        if st.session_state.get("sudden_change_active", False):
            active_alerts.append(st.session_state.sudden_change_alert_text)

    # Publish only alert-state transitions, not every 1–2 second fragment refresh.
    # Stable keys prevent changing temperature text from creating duplicate alerts.
    current_ntfy_alerts = {}
    if connected and st.session_state.last_reading_time is not None:
        current_temperature = float(reading["temperature"])
        if current_temperature > float(st.session_state.high_temperature_threshold):
            current_ntfy_alerts["high_temperature"] = (
                f"Temperature is {current_temperature:.1f} °C; configured limit is "
                f"{float(st.session_state.high_temperature_threshold):g} °C."
            )
        if st.session_state.get("sudden_change_active", False):
            current_ntfy_alerts["sudden_change"] = str(
                st.session_state.get("sudden_change_alert_text", "Sudden temperature change detected.")
            )

    previous_ntfy_alerts = st.session_state.ntfy_previous_alerts
    if st.session_state.ntfy_enabled and st.session_state.ntfy_topic.strip():
        # A valid connected sample is required before declaring an alert recovered.
        for alert_key, alert_message in current_ntfy_alerts.items():
            if alert_key not in previous_ntfy_alerts:
                zone_name = st.session_state.get("selected_plant_zone", "Selected zone")
                zone_id = plant_zones.get(zone_name, "")
                notification_text = (
                    f"⚠ SABOTAGE\nZone: {zone_name} ({zone_id})\n"
                    f"{alert_message}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                    "SENTINEL environmental monitoring system"
                )
                ok, detail = send_ntfy_notification(
                    st.session_state.ntfy_topic,
                    "SENTINEL: SABOTAGE ALERT",
                    notification_text,
                    priority="urgent" if alert_key == "high_temperature" else "high",
                    tags="rotating_light,warning",
                )
                if ok:
                    st.session_state.ntfy_last_error = ""
                    st.session_state.ntfy_last_sent = f"Alert notification sent: {alert_key.replace('_', ' ')}"
                else:
                    st.session_state.ntfy_last_error = detail

        if connected and st.session_state.last_reading_time is not None:
            for alert_key, old_message in previous_ntfy_alerts.items():
                if alert_key not in current_ntfy_alerts:
                    zone_name = st.session_state.get("selected_plant_zone", "Selected zone")
                    zone_id = plant_zones.get(zone_name, "")
                    recovery_text = (
                        f"✓ ALERT RESOLVED\nZone: {zone_name} ({zone_id})\n"
                        f"Resolved condition: {old_message}\n"
                        f"Current temperature: {float(reading['temperature']):.1f} °C\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
                        "SENTINEL has returned to normal for this alert condition."
                    )
                    ok, detail = send_ntfy_notification(
                        st.session_state.ntfy_topic,
                        "SENTINEL: ALERT RESOLVED",
                        recovery_text,
                        priority="default",
                        tags="white_check_mark,seedling",
                    )
                    if ok:
                        st.session_state.ntfy_last_error = ""
                        st.session_state.ntfy_last_sent = f"Recovery notification sent: {alert_key.replace('_', ' ')}"
                    else:
                        st.session_state.ntfy_last_error = detail

    # Keep the last known alert state while disconnected, so a USB interruption
    # cannot generate a misleading recovery message.
    if connected and st.session_state.last_reading_time is not None:
        st.session_state.ntfy_previous_alerts = current_ntfy_alerts.copy()

    if active_alerts:
        st.markdown(ALERT_STYLE, unsafe_allow_html=True)
        st.markdown(
            '<div class="sentinel-alert-overlay"></div>'
            '<div class="sentinel-alert-banner"><h2>⚠ SABOTAGE</h2>'
            + ''.join(f'<p>{msg}</p>' for msg in active_alerts)
            + '</div>',
            unsafe_allow_html=True,
        )

    # Invisible persistent browser audio. The visible red sound-control box is
    # removed; the first ordinary page interaction unlocks audio where the browser
    # requires a user gesture. Once unlocked, alerts start/stop automatically.
    alarm_is_active = bool(active_alerts) and alarm_sound_enabled
    st.components.v1.html(f"""
    <html><body style="margin:0;padding:0;height:0;overflow:hidden">
      <script>
      (() => {{
        const soundUrl = "https://www.myinstants.com/media/sounds/among-us-reactor-meldown-sound-effect-128-kbps-abdwap2.mp3";
        const active = {str(alarm_is_active).lower()};
        try {{
          const doc = window.parent.document;
          let audio = doc.getElementById('sentinel-reactor-meltdown-audio');
          if (!audio) {{
            audio = doc.createElement('audio');
            audio.id = 'sentinel-reactor-meltdown-audio';
            audio.src = soundUrl;
            audio.preload = 'auto';
            audio.loop = true;
            audio.volume = 1.0;
            doc.body.appendChild(audio);
          }}
          audio.loop = true;
          audio.dataset.sentinelAlarmActive = active ? '1' : '0';
          if (active) {{
            const playPromise = audio.play();
            if (playPromise && playPromise.catch) playPromise.catch(() => {{}});
          }} else {{
            audio.pause();
            audio.currentTime = 0;
          }}
          if (!doc.documentElement.dataset.sentinelAudioUnlockBound) {{
            doc.documentElement.dataset.sentinelAudioUnlockBound = '1';
            const unlockAudio = () => {{
              const player = doc.getElementById('sentinel-reactor-meltdown-audio');
              if (player && player.dataset.sentinelAlarmActive === '1') {{
                player.loop = true;
                const promise = player.play();
                if (promise && promise.catch) promise.catch(() => {{}});
              }}
            }};
            doc.addEventListener('pointerdown', unlockAudio, true);
            doc.addEventListener('keydown', unlockAudio, true);
          }}
        }} catch (e) {{}}
      }})();
      </script>
    </body></html>
    """, height=0)

    if page == "Overview":
        c1, c2, c3 = st.columns(3, gap="medium")
        cards = [
            (c1, "♨", "Temperature", f"{reading['temperature']:.1f}", "°C", "#e38baa"),
            (c2, "💧", "Humidity", f"{reading['humidity']:.1f}", "%", "#82c99b"),
            (
                c3, "☀", "Light Sensor",
                classify_light_level(reading["light"]),
                "" if reading["light"] == 0 else f"ADC {reading['light']}", "#e9c16f"
            ),
        ]
        for col, icon, label, value, unit, color in cards:
            with col:
                st.markdown(
                    f'<div class="panel"><div class="metric-icon" style="color:{color}">{icon}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}<span style="font-size:17px"> {unit}</span></div>'
                    '<div class="muted">Current sensor reading</div></div>',
                    unsafe_allow_html=True,
                )

        chart_col, plant_col = st.columns([2.25, 1], gap="medium")
        session_df = pd.DataFrame(st.session_state.session_history)
        with chart_col:
            st.markdown(
                '<div class="panel"><div class="section-title">Temperature History</div>',
                unsafe_allow_html=True,
            )

            # Reload the CSV on every fragment refresh. The fragment reruns
            # this chart at the selected interval as fresh sensor samples arrive.
            history_df = load_history().copy()
            if not history_df.empty:
                history_df["temperature"] = pd.to_numeric(
                    history_df["temperature"], errors="coerce"
                )
                history_df = history_df.dropna(subset=["temperature"])

            if not history_df.empty:
                # A true scrolling strip chart: keep the horizontal axis fixed at
                # 1..60. Each new reading is placed at the right edge and the
                # oldest visible reading falls off the left edge.
                temperatures = history_df["temperature"].astype(float).tolist()

                if len(temperatures) >= 3:
                    rolling = calculate_rolling_average(
                        temperatures, window_size=3
                    )
                    history_df["rolling_average"] = [None, None] + rolling
                else:
                    history_df["rolling_average"] = [None] * len(history_df)

                visible = history_df.tail(60).copy().reset_index(drop=True)
                visible["plot_position"] = range(1, len(visible) + 1)
                latest = visible.iloc[-1]
                latest_x = int(latest["plot_position"])
                latest_temp = float(latest["temperature"])
                latest_average = latest["rolling_average"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=visible["plot_position"],
                    y=visible["temperature"],
                    mode="lines+markers",
                    name="Raw temperature",
                    opacity=1,
                    line=dict(color="#55B4E5", width=2.5),
                    marker=dict(size=4, color="#55B4E5"),
                    hovertemplate="Recent sample %{x}<br>Temperature: %{y:.1f} °C<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=visible["plot_position"],
                    y=visible["rolling_average"],
                    mode="lines",
                    name="3-reading average",
                    opacity=1,
                    line=dict(color="#F2B766", width=2.5),
                    connectgaps=False,
                    hovertemplate="Recent sample %{x}<br>3-reading average: %{y:.1f} °C<extra></extra>",
                ))

                # A prominent marker and value label identify the newest sample.
                fig.add_trace(go.Scatter(
                    x=[latest_x], y=[latest_temp],
                    mode="markers+text",
                    name="Latest reading",
                    opacity=1,
                    text=[f"{latest_temp:.1f} °C"],
                    textposition="top center",
                    # Allow the newest value label to extend into the reserved margin.
                    cliponaxis=False,
                    textfont=dict(color="#FFFFFF", size=13),
                    marker=dict(color="#55B4E5", size=13, line=dict(color="#FFFFFF", width=2)),
                    showlegend=False,
                    hovertemplate="LATEST READING<br>%{y:.1f} °C<extra></extra>",
                ))

                if pd.notna(latest_average):
                    fig.add_trace(go.Scatter(
                        x=[latest_x],
                        y=[float(latest_average)],
                        mode="markers",
                        name="Latest average point",
                        opacity=1,
                        marker=dict(color="#F2B766", size=10, line=dict(color="#202624", width=1)),
                        showlegend=False,
                        hovertemplate="Latest 3-reading average: %{y:.1f} °C<extra></extra>",
                    ))

                fig.update_layout(
                    height=330,
                    # Reserve space at the right for the newest marker and its label.
                    margin=dict(l=48, r=58, t=34, b=48),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,15,20,0.72)",
                    font=dict(color="#D8DED9", family="Roboto, sans-serif", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    hovermode="x unified",
                    xaxis=dict(
                        title="Recent readings · newest at right",
                        # The newest sample is at x=60; four extra units provide
                        # visible breathing room before the right chart border.
                        range=[1, 64],
                        dtick=5,
                        showgrid=False,
                        zeroline=False,
                        fixedrange=True,
                    ),
                    yaxis=dict(
                        title="Temperature (°C)",
                        rangemode="normal",
                        gridcolor="rgba(180,195,190,0.15)",
                        zeroline=False,
                    ),
                    # Keep this stable across readings: changing it per timestamp
                    # forces Plotly to reset its chart state on every fragment run.
                    uirevision="sentinel-temperature-live",
                    # Interpolate new trace data rather than snapping instantly.
                    transition=dict(
                        duration=150,  # 0.15 seconds
                        easing="linear",
                    ),
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="sentinel_live_temperature_chart",
                )
                updated_at = latest["timestamp"]
                st.caption(
                    f"LATEST: {latest_temp:.1f} °C · "
                    f"Updated: {updated_at.strftime('%H:%M:%S') if pd.notna(updated_at) else '—'} · "
                    f"Showing {len(visible)} latest readings. "
                    "New readings enter on the right; older readings slide left."
                )
            else:
                st.info("Waiting for saved readings.")

            st.markdown("</div>", unsafe_allow_html=True)

        with plant_col:
            st.markdown("""
            <div class="panel">
                <div class="section-title">Plant Comfort</div>
                <div class="plant-illustration"><div class="plant-scene">
                    <div class="plant-stem"></div><div class="plant-leaf leaf-left"></div>
                    <div class="plant-leaf leaf-right"></div><div class="plant-leaf leaf-small"></div>
                    <div class="plant-pot"></div><div class="plant-pot-rim"></div>
                </div></div>
                <div class="comfort-status">Monitoring Active</div>
                <div class="comfort-description">Environmental readings are being monitored.
                The fine-tuned SENTINEL image classifier is available in the live webcam panel below.</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Recent Readings")
        if not session_df.empty:
            st.dataframe(session_df.tail(8).iloc[::-1], use_container_width=True, hide_index=True)

        render_webcam_ai_panel()

    elif page == "Live Sensors":
        st.subheader("Live Sensor Readings")
        a, b, c = st.columns(3)
        a.metric("Temperature", f"{reading['temperature']:.1f} °C")
        b.metric("Humidity", f"{reading['humidity']:.1f} %")
        light_display = (
            "0" if reading["light"] == 0
            else f"{classify_light_level(reading['light'])} ADC {reading['light']}"
        )
        c.metric(
            "Light Level",
            light_display,
            help=f"Raw LDR ADC reading: {reading['light']} (0-1023)"
        )
        st.caption("Light is the Arduino Uno raw 10-bit ADC value, not lux.")

    elif page == "History":
        st.subheader("Environmental History")
        history = load_history()
        if history.empty:
            st.info("No saved readings yet.")
        else:
            st.line_chart(history.set_index("timestamp")[["temperature", "humidity"]].tail(500), height=350, use_container_width=True)
            st.dataframe(history.tail(100).iloc[::-1], use_container_width=True, hide_index=True)

    elif page == "Alerts":
        st.subheader("Environmental Alerts")
        if not connected or st.session_state.last_reading_time is None:
            st.info("Arduino sensor data is unavailable. Alerts will be evaluated when valid readings arrive.")
        elif active_alerts:
            for alert in active_alerts:
                st.error(alert)
            st.caption("Sudden-change alerts compare consecutive valid live readings; they are not calculated from the smoothed chart average.")
        else:
            st.success("No active temperature alerts.")
        st.caption(f"High temperature: above {float(st.session_state.high_temperature_threshold):g} °C. Sudden rise or fall: absolute change greater than 5 °C between consecutive valid readings. The reactor alarm remains active until temperature is normal and a stable valid sample is received. These are prototype thresholds, not validated plant-specific safety limits.")


if page == "Settings":
    st.subheader("System Settings")
    st.markdown("#### Alert Configuration")
    st.caption(
        "Set the alert limit, then click Apply. The selected value is written to "
        "storage/sentinel_settings.json and remains in effect after dashboard reruns."
    )

    # Use a form so the one-second sensor fragment refresh cannot interrupt
    # an in-progress edit. The committed value is persisted before rerunning.
    if "high_temperature_threshold_input" not in st.session_state:
        st.session_state.high_temperature_threshold_input = float(
            st.session_state.high_temperature_threshold
        )

    with st.form("sentinel_temperature_threshold_form", clear_on_submit=False):
        st.number_input(
            "High-temperature alert threshold (°C)",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            key="high_temperature_threshold_input",
            help="Alerts activate when a valid temperature reading rises above this value.",
        )
        apply_threshold = st.form_submit_button("Apply threshold")

    if apply_threshold:
        selected_threshold = float(st.session_state.high_temperature_threshold_input)
        st.session_state.high_temperature_threshold = selected_threshold
        try:
            save_app_settings({"high_temperature_threshold": selected_threshold})
            st.session_state.temperature_threshold_save_error = ""
            st.success(f"Threshold saved: {selected_threshold:g} °C")
        except (OSError, ValueError, TypeError) as exc:
            st.session_state.temperature_threshold_save_error = str(exc)
            st.error(f"Could not save the temperature threshold: {exc}")

    st.caption(
        f"Active high-temperature alert threshold: "
        f"{float(st.session_state.high_temperature_threshold):g} °C"
    )
    if st.session_state.get("temperature_threshold_save_error"):
        st.error(
            "Could not save the temperature threshold: "
            f"{st.session_state.temperature_threshold_save_error}"
        )

    st.divider()
    st.markdown("#### Phone Push Notifications (ntfy)")
    st.caption(
        "Create a private topic in the ntfy app on your phone, subscribe to the same topic, "
        "then enter it here. Treat the topic name like a password: anyone who knows it "
        "may be able to publish to or subscribe to its notifications."
    )
    if "ntfy_topic_input" not in st.session_state:
        st.session_state.ntfy_topic_input = st.session_state.ntfy_topic
    if "ntfy_enabled_input" not in st.session_state:
        st.session_state.ntfy_enabled_input = st.session_state.ntfy_enabled

    with st.form("sentinel_ntfy_settings_form", clear_on_submit=False):
        st.text_input(
            "Private ntfy topic",
            key="ntfy_topic_input",
            placeholder="Example: sentinel-plant-a-unique-random-name",
            help="Use a hard-to-guess topic name, 8–64 characters: letters, numbers, hyphens or underscores.",
        )
        st.checkbox("Enable phone notifications", key="ntfy_enabled_input")
        save_ntfy_settings = st.form_submit_button("Save ntfy settings")

    if save_ntfy_settings:
        selected_topic = str(st.session_state.ntfy_topic_input or "").strip()
        selected_enabled = bool(st.session_state.ntfy_enabled_input)
        if selected_enabled and not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", selected_topic):
            st.error("Enter a valid private topic (8–64 letters, numbers, hyphens or underscores) before enabling notifications.")
        else:
            st.session_state.ntfy_topic = selected_topic
            st.session_state.ntfy_enabled = selected_enabled
            # Re-evaluate active conditions against the newly saved destination.
            st.session_state.ntfy_previous_alerts = {}
            try:
                save_app_settings({"ntfy_topic": selected_topic, "ntfy_enabled": selected_enabled})
                st.session_state.ntfy_last_error = ""
                st.success("ntfy settings saved.")
            except OSError as exc:
                st.session_state.ntfy_last_error = str(exc)
                st.error(f"Could not save ntfy settings: {exc}")

    st.write("Phone notifications:", "Enabled" if st.session_state.ntfy_enabled else "Disabled")
    st.write("ntfy delivery status:", st.session_state.ntfy_last_sent)
    if st.button("Send test phone notification", disabled=not (st.session_state.ntfy_enabled and st.session_state.ntfy_topic.strip())):
        ok, detail = send_ntfy_notification(
            st.session_state.ntfy_topic,
            "SENTINEL: TEST NOTIFICATION",
            "Test successful. Your phone is subscribed to SENTINEL notifications.",
            priority="default",
            tags="white_check_mark,seedling",
        )
        if ok:
            st.session_state.ntfy_last_error = ""
            st.session_state.ntfy_last_sent = "Test notification sent successfully."
            st.success("Test notification sent. Check the ntfy app on your phone.")
        else:
            st.session_state.ntfy_last_error = detail
            st.error(f"Test notification failed: {detail}")
    if st.session_state.ntfy_last_error:
        st.warning(f"Last ntfy delivery/configuration error: {st.session_state.ntfy_last_error}")
    st.divider()
    st.write("Data source: Arduino Uno (USB serial)")
    st.write("Firmware management: Automatic via Arduino CLI (no Arduino IDE window required)")
    st.write("Firmware status:", st.session_state.get("arduino_firmware_status", "Not started"))
    st.write("Selected port:", port_choice)
    st.write("Detected ports:", ", ".join(item.device for item in get_available_serial_ports()) or "None")
    settings_connected = (
        st.session_state.get("last_reading_time") is not None
        and st.session_state.get("serial_error") is None
    )
    st.write("Connection status:", "Connected" if settings_connected else "Disconnected / waiting for data")
    st.write("Update interval:", refresh_rate)
    st.write("CSV storage file:", str(CSV))
    st.write("Background image:", str(BG))
    st.caption("Replace assets/plant_bg.jpg to change the background image.")
    if st.button("Clear current session graph data"):
        st.session_state.session_history = []
        st.success("Session graph cleared. Saved CSV history was not deleted.")
else:
    live_panel()
