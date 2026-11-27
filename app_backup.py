import base64
import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import serial
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

from processing.analyzer import calculate_rolling_average

BASE = Path(__file__).resolve().parent
BG = BASE / "assets" / "plant_bg.jpg"
CSV = BASE / "storage" / "dashboard_readings.csv"
AI_ROOT = BASE / "processing" / "SENTINEL_AI"
AI_MODEL_PATH = AI_ROOT / "models" / "sentinel_mobilenetv3_small_finetuned_best.pth"
MAX_CSV_ROWS = 10_000
RETENTION_CHECK_INTERVAL = 100

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
        "The webcam frame is classified by your fine-tuned MobileNetV3-Small "
        "model. The model was trained on PlantVillage images; webcam results "
        "are experimental and are not a confirmed plant diagnosis."
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

                        st.markdown(
                            f'<div class="muted">TOP MODEL PREDICTION</div>'
                            f'<div style="font-size:20px;font-weight:700;color:#f0eadd;margin:8px 0">'
                            f'{display_name}</div>',
                            unsafe_allow_html=True
                        )
                        st.metric("Model score", f"{top_score * 100:.1f}%")

                        st.markdown("**Top 3 classes**")
                        for rank, (name, score) in enumerate(predictions, start=1):
                            readable = name.replace("___", " | ").replace("_", " ")
                            st.write(f"{rank}. {readable} — {score * 100:.1f}%")

                        st.caption(
                            f"Checkpoint: fine-tuned model · Epoch {checkpoint.get('epoch', 'unknown')} · "
                            f"{len(predictions)} leading classes shown"
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
        "Important: the displayed score is the model's softmax score, not a "
        "measure of real-world diagnostic certainty. PlantVillage test accuracy "
        "does not guarantee webcam accuracy, especially with different lighting, "
        "backgrounds, plant species or camera angles."
    )


def simulated_reading():
    old = st.session_state.simulation
    current = {
        "temperature": round(max(15, min(40, old["temperature"] + random.uniform(-.7, .7))), 1),
        "humidity": round(max(20, min(100, old["humidity"] + random.uniform(-1.8, 1.8))), 1),
        "light": int(max(0, min(1023, old["light"] + random.randint(-28, 28)))),
    }
    st.session_state.simulation = current
    return current


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


@st.cache_resource(show_spinner=False)
def get_serial_connection(port, baud_rate=9600):
    """Keep one serial connection open instead of reopening it on every refresh."""
    return serial.Serial(port, baud_rate, timeout=0.25)


def read_available_arduino_readings(port):
    """Read all complete lines currently queued on the persistent serial port."""
    device = get_serial_connection(port)
    lines = []

    # Wait briefly for the next line. Do NOT reset the input buffer here:
    # resetting it discards readings that arrived between dashboard refreshes.
    first_line = device.readline().decode("utf-8", errors="ignore").strip()
    if first_line:
        lines.append(first_line)

    # Drain any additional complete lines already waiting, so delayed UI
    # refreshes do not silently throw away samples.
    while device.in_waiting:
        extra = device.readline().decode("utf-8", errors="ignore").strip()
        if not extra:
            break
        lines.append(extra)

    readings = []
    for line in lines:
        try:
            readings.append(parse_sensor_line(line))
        except (ValueError, TypeError):
            # Ignore incomplete boot messages or malformed serial lines.
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
if "simulation" not in st.session_state:
    st.session_state.simulation = {"temperature": 22.0, "humidity": 65.0, "light": 600}
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = None
if "last_reading_time" not in st.session_state:
    st.session_state.last_reading_time = None
if "serial_error" not in st.session_state:
    st.session_state.serial_error = None
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
        "⌂　 Overview": "Overview",
        "⌁　 Live Sensors": "Live Sensors",
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
    st.selectbox("Plant zone", ["Greenhouse A"], label_visibility="collapsed")
    st.caption("ID: GH-A-001")
    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)

    source = st.selectbox("Data Source", ["Simulation", "Arduino USB"], key="data_source")
    port = st.text_input(
        "Arduino Port", "/dev/ttyACM0", key="arduino_port",
        disabled=(source != "Arduino USB")
    )
    refresh_rate = st.selectbox("Dashboard update rate", ["1 second", "2 seconds"], index=0)
    refresh_seconds = 1 if refresh_rate == "1 second" else 2
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("SENTINEL Local Prototype")


# ---------------------------- Static page heading ----------------------------
st.title("Plant Environment Overview")
st.caption("Live values update independently; the dashboard layout stays in place.")


# ---------------------------- Partial live update ----------------------------
@st.fragment(run_every=f"{refresh_seconds}s")
def live_panel():
    """Only this region reruns on the timer; sidebar and layout remain still."""
    new_readings = []
    error = None

    if source == "Simulation":
        new_readings = [simulated_reading()]
    else:
        try:
            new_readings = read_available_arduino_readings(port)
            st.session_state.serial_error = None
        except (serial.SerialException, OSError, ValueError) as exc:
            error = str(exc)
            st.session_state.serial_error = error
            # Release a failed connection so the next fragment run can reconnect.
            try:
                get_serial_connection.clear()
            except Exception:
                pass

    if new_readings:
        now = datetime.now()
        for offset, item in enumerate(new_readings):
            record = {"timestamp": now, **item}
            st.session_state.session_history.append(record)
        st.session_state.session_history = st.session_state.session_history[-1000:]
        st.session_state.latest_reading = new_readings[-1]
        st.session_state.last_reading_time = now
        save_readings(new_readings)

    reading = st.session_state.latest_reading

    if error:
        st.error(f"Arduino connection problem: {error}")
    elif source == "Arduino USB" and reading is None:
        st.info("Waiting for the first Arduino reading. Check that the sketch sends CSV lines at 9600 baud.")
    elif source == "Arduino USB" and not new_readings:
        st.caption("Connection active · waiting for next sensor sample")

    if reading is not None:
        label = "SIMULATED DATA" if source == "Simulation" else "LIVE ARDUINO CONNECTION"
        updated = st.session_state.last_reading_time
        st.markdown(
            f'<div class="status-line">● {label}</div>'
            f'<div class="muted">Last updated: {updated.strftime("%I:%M:%S %p") if updated else "—"}</div>',
            unsafe_allow_html=True,
        )

    if page == "Overview":
        if reading is None:
            st.info("Select Simulation or connect the Arduino to display readings.")
            return

        c1, c2, c3 = st.columns(3, gap="medium")
        cards = [
            (c1, "♨", "Temperature", f"{reading['temperature']:.1f}", "°C", "#e38baa"),
            (c2, "💧", "Humidity", f"{reading['humidity']:.1f}", "%", "#82c99b"),
            (c3, "☀", "Light Sensor", str(reading["light"]), "", "#e9c16f"),
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
        if reading is None:
            st.info("Waiting for sensor data.")
        else:
            a, b, c = st.columns(3)
            a.metric("Temperature", f"{reading['temperature']:.1f} °C")
            b.metric("Humidity", f"{reading['humidity']:.1f} %")
            c.metric("Light ADC", reading["light"])
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
        if reading is None:
            st.info("Waiting for sensor data.")
        else:
            alerts = []
            if reading["temperature"] >= 35:
                alerts.append("High temperature reading (demo threshold: 35 °C).")
            if reading["humidity"] >= 80:
                alerts.append("High humidity reading (demo threshold: 80%).")
            if reading["light"] < 300:
                alerts.append("Low light ADC reading (demo threshold: 300).")
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("No configured threshold alerts right now.")
            st.caption("These are demonstration thresholds, not validated plant-specific stress limits.")


if page == "Settings":
    st.subheader("System Settings")
    st.write("Data source:", source)
    st.write("Arduino serial port:", port if source == "Arduino USB" else "Not in use")
    st.write("Update interval:", refresh_rate)
    st.write("CSV storage file:", str(CSV))
    st.write("Background image:", str(BG))
    st.caption("Replace assets/plant_bg.jpg to change the background image.")
    if st.button("Clear current session graph data"):
        st.session_state.session_history = []
        st.success("Session graph cleared. Saved CSV history was not deleted.")
else:
    live_panel()
