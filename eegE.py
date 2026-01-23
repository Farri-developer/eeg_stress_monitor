import time, csv, os, sys, subprocess
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_byprop

# =========================
# SESSION FOLDER
# =========================
session_id = input("Enter Session Number (e.g. 1, 2, 101): ").strip()
if not session_id.isdigit():
    raise ValueError("❌ Session must be a number")

folder = f"Session_{session_id}"
os.makedirs(folder, exist_ok=True)

# =========================
# FILE PATHS
# =========================
raw_file = f"{folder}/raw_eeg.csv"
gui_file = f"{folder}/gui_1s.csv"

def init_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

init_csv(raw_file, ["timestamp","ch1","ch2","ch3","ch4"])
init_csv(gui_file, ["timestamp","Delta","Theta","Alpha","Beta","Gamma",
                    "Stress_Index","Stress_Label","Trend","User_Comment"])

# =========================
# START MUSE STREAM
# =========================
proc = subprocess.Popen([sys.executable, "-m", "muselsl", "stream"])
print("🎧 Waiting for Muse...")

streams = resolve_byprop("type", "EEG", timeout=15)
if not streams:
    proc.terminate()
    raise RuntimeError("❌ Muse not found")

inlet = StreamInlet(streams[0])
fs = int(inlet.info().nominal_srate())
channels = inlet.info().channel_count()
print("✅ Muse Connected | FS:", fs)

# =========================
# CONFIG
# =========================
GUI_SEC = 1
GUI_SAMPLES = int(fs * GUI_SEC)
STREAM_TIMEOUT = 30

bands = {
    "Delta": (1,4),
    "Theta": (4,8),
    "Alpha": (8,13),
    "Beta": (13,30),
    "Gamma": (30,50)
}
band_names = list(bands.keys())

# =========================
# BUFFERS
# =========================
gui_buffer = np.zeros((channels, GUI_SAMPLES))
last_sample_time = time.time()
stress_hist = []
last_gui = time.time()
last_stress = None  # for trend

# Store user comment temporarily
user_comment = ""

# =========================
# SIGNAL PROCESSING
# =========================
def band_powers(sig):
    sig = np.array(sig)
    if sig.ndim == 1:
        sig = sig.reshape(1, -1)
    powers = {b: 0.0 for b in bands}
    for ch in sig:
        ch = ch - np.mean(ch)
        fft = np.abs(np.fft.rfft(ch))**2
        freqs = np.fft.rfftfreq(len(ch), 1/fs)
        total = np.sum(fft)
        for b, (f1, f2) in bands.items():
            powers[b] += np.sum(fft[(freqs >= f1) & (freqs <= f2)]) / total if total else 0
    for b in powers:
        powers[b] /= sig.shape[0]
    return powers

def stress_index(b, a, t):
    return b / (a + t + 1e-6)

def stress_label(v):
    if v < 0.5: return "L"
    elif v < 1.5: return "M"
    else: return "H"

# =========================
# GUI SETUP
# =========================
app = QtWidgets.QApplication([])
win = QtWidgets.QMainWindow()
win.setWindowTitle(f"NeuroMoodTracker – Session {session_id}")

raw_plot = pg.PlotWidget(title="Raw EEG (1s)")
raw_curve = raw_plot.plot(pen='g')
stress_plot = pg.PlotWidget(title="Stress Index")
stress_curve = stress_plot.plot(pen=pg.mkPen('r', width=3))

# Comment input
comment_input = QtWidgets.QLineEdit()
comment_input.setPlaceholderText("Type your comment here...")

# Submit button
submit_btn = QtWidgets.QPushButton("Add Comment")

def submit_comment():
    global user_comment
    user_comment = comment_input.text().strip()
    comment_input.clear()
    print(f"💬 User Comment Added: {user_comment}")

submit_btn.clicked.connect(submit_comment)

logout_btn = QtWidgets.QPushButton("Logout / Stop")
logout_btn.setStyleSheet("font-size:16px; padding:10px;")
logout_btn.setFixedWidth(150)

def logout():
    timer.stop()
    proc.terminate()
    print("🛑 Recording Stopped")
    app.quit()

logout_btn.clicked.connect(logout)

layout = QtWidgets.QGridLayout()
layout.addWidget(raw_plot, 0, 0)
layout.addWidget(stress_plot, 0, 1)
layout.addWidget(comment_input, 1, 0)
layout.addWidget(submit_btn, 1, 1)
layout.addWidget(logout_btn, 2, 0, 1, 2, alignment=QtCore.Qt.AlignCenter)
central = QtWidgets.QWidget()
central.setLayout(layout)
win.setCentralWidget(central)

# =========================
# MAIN LOOP
# =========================
def update():
    global gui_buffer, last_gui, last_sample_time, stress_hist, last_stress, user_comment

    sample, ts = inlet.pull_sample(timeout=0.5)
    if sample:
        if np.all(np.array(sample[:channels]) == 0):
            return

        last_sample_time = time.time()
        # Save raw EEG
        with open(raw_file,"a",newline="") as f:
            csv.writer(f).writerow([ts] + sample[:channels])

        gui_buffer = np.roll(gui_buffer, -1, axis=1)
        gui_buffer[:, -1] = sample[:channels]
        raw_curve.setData(gui_buffer[0])

    # Stream timeout
    if time.time() - last_sample_time > STREAM_TIMEOUT:
        print(f"❌ No EEG data for {STREAM_TIMEOUT}s. Closing app...")
        logout()
        return

    # GUI update every 1 second
    if time.time() - last_gui >= GUI_SEC:
        pct = band_powers(gui_buffer)
        s_val = stress_index(pct["Beta"], pct["Alpha"], pct["Theta"])
        label = stress_label(s_val)

        # Determine trend comment
        if last_stress is None:
            trend = "Stable"
        elif s_val > last_stress + 0.02:
            trend = "Increasing"
        elif s_val < last_stress - 0.02:
            trend = "Decreasing"
        else:
            trend = "Stable"

        last_stress = s_val

        stress_hist.append(s_val)
        color = 'r' if trend == "Increasing" else ('g' if trend == "Decreasing" else 'y')
        stress_curve.setPen(pg.mkPen(color, width=3))
        stress_curve.setData(stress_hist)

        # Save GUI CSV including user comment
        with open(gui_file,"a",newline="") as f:
            csv.writer(f).writerow([time.time()] + [pct[b] for b in band_names] + [s_val, label, trend, user_comment])

        print(f"🖥 GUI update | Stress: {label} ({s_val:.3f}) | {trend} | User Comment: {user_comment}")

        # Clear user comment after saving
        user_comment = ""
        last_gui = time.time()

# =========================
# TIMER
# =========================
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

print("🚀 Recording Started")
win.show()
app.exec()
