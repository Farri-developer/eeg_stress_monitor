import time, csv, os, sys, subprocess
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import find_peaks

# ===== CROSS-PLATFORM HEARTBEAT BEEP =====
try:
    import simpleaudio as sa

    beep_wave = (np.sin(2 * np.pi * np.arange(0, 0.1, 1 / 44100) * 1000) * 32767).astype(np.int16)
    beep_obj = sa.WaveObject(beep_wave, 1, 2, 44100)
except:
    beep_obj = None
    print("⚠ simpleaudio not installed, beep disabled")

# ===== SESSION SETUP =====
session_id = input("Enter Session Number: ").strip()
if not session_id.isdigit():
    raise ValueError("Session must be a number")
folder = f"PPG_Session_{session_id}"
os.makedirs(folder, exist_ok=True)

# Dataset files
dataset_file = f"{folder}/dataset.csv"

raw_ppg_file = f"{folder}/raw_ppg.csv"

if not os.path.exists(dataset_file):
    with open(dataset_file, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "HR", "HRV", "Stress_Index"])

if not os.path.exists(raw_ppg_file):
    with open(raw_ppg_file, "w", newline="") as f:
        header = ["timestamp"] + [f"PPG_Ch{i + 1}" for i in range(4)] + ["HR"]
        csv.writer(f).writerow(header)

# ===== START MUSE PPG =====
proc = subprocess.Popen([sys.executable, "-m", "muselsl", "stream", "--ppg"])
print("🎧 Waiting for Muse PPG...")
streams = resolve_byprop("type", "PPG", timeout=15)
if not streams:
    proc.terminate()
    raise RuntimeError("❌ Muse PPG not found")
inlet = StreamInlet(streams[0])
fs = int(inlet.info().nominal_srate())
channels = inlet.info().channel_count()
print("✅ Muse PPG Connected | FS:", fs)

# ===== CONFIG =====
ROLLING_SEC = 10  # Show 10s of data for better ECG-style view
GUI_SEC = 1  # compute HR/HRV every second
ROLLING_SAMPLES = int(fs * ROLLING_SEC)
STREAM_TIMEOUT = 30

# ===== BUFFERS =====
ppg_buffer = np.zeros((channels, ROLLING_SAMPLES))
hr_buffer = np.zeros(ROLLING_SAMPLES)
last_sample_time = time.time()
last_gui = time.time()
last_peaks = []
hr_history = []
hrv_history = []
peak_labels = []  # Store text items for peak labels


# ===== PPG -> HR/HRV =====
def compute_hr_hrv(sig, fs):
    ppg = sig[0] - np.mean(sig[0])
    # Normalize for better peak detection
    ppg = ppg / (np.std(ppg) + 1e-6)
    peaks, _ = find_peaks(ppg, distance=fs * 0.4, prominence=0.3)
    if len(peaks) < 2:
        return 0, 0, peaks
    rr_intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    hrv = np.std(rr_intervals)
    return hr, hrv, peaks


def stress_index(hr, hrv):
    return hr / (hrv + 1e-6)


# ===== GUI SETUP =====
app = QtWidgets.QApplication([])
win = QtWidgets.QMainWindow()
win.setWindowTitle(f"PPG Monitor – Session {session_id}")
win.resize(1400, 800)

# ===== PPG PLOT (ECG-STYLE) =====
ppg_plot = pg.PlotWidget(title="PPG Signal (ECG-Style)")
ppg_plot.setBackground('w')  # White background like ECG paper
ppg_plot.showGrid(x=True, y=True, alpha=0.4)
ppg_plot.getAxis('bottom').setPen(pg.mkPen('#CCCCCC', width=1))
ppg_plot.getAxis('left').setPen(pg.mkPen('#CCCCCC', width=1))
ppg_plot.setYRange(-2, 3)
ppg_curve = ppg_plot.plot(pen=pg.mkPen('#E53935', width=2))  # Red like ECG
peak_scatter = pg.ScatterPlotItem(size=14, pen=pg.mkPen('#D32F2F', width=2),
                                  brush=pg.mkBrush(255, 0, 0, 100), symbol='o')
ppg_plot.addItem(peak_scatter)

# ===== HR PLOT =====
hr_plot = pg.PlotWidget(title="Heart Rate (BPM)")
hr_plot.setBackground('#1a1a1a')
hr_plot.showGrid(x=True, y=True, alpha=0.3)
hr_plot.setYRange(40, 180)
hr_curve = hr_plot.plot(pen=pg.mkPen('#00E676', width=2))
hr_peak_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('#FF6B6B'), brush=pg.mkBrush('#FF6B6B'))
hr_plot.addItem(hr_peak_scatter)

# ===== METRICS LAYOUT =====
metrics_layout = QtWidgets.QGridLayout()
hr_label = QtWidgets.QLabel("HR: 0 bpm")
hr_label.setStyleSheet("font-size: 28px; color: #00FF00; font-weight:bold; background:#1a1a1a; padding:10px;")
hrv_label = QtWidgets.QLabel("HRV: 0 s")
hrv_label.setStyleSheet("font-size: 28px; color: #FF4500; font-weight:bold; background:#1a1a1a; padding:10px;")
stress_label = QtWidgets.QLabel("Stress: 0")
stress_label.setStyleSheet("font-size: 28px; color: #1E90FF; font-weight:bold; background:#1a1a1a; padding:10px;")
metrics_layout.addWidget(hr_label, 0, 0)
metrics_layout.addWidget(hrv_label, 0, 1)
metrics_layout.addWidget(stress_label, 0, 2)

# ===== STOP BUTTON =====
logout_btn = QtWidgets.QPushButton("Stop Recording")
logout_btn.setStyleSheet("font-size: 18px; padding: 12px; background: #FF5555; color: white; font-weight:bold;")

# ===== MAIN LAYOUT =====
main_layout = QtWidgets.QVBoxLayout()
main_layout.addWidget(ppg_plot)
main_layout.addWidget(hr_plot)
main_layout.addLayout(metrics_layout)
main_layout.addWidget(logout_btn)
central = QtWidgets.QWidget()
central.setLayout(main_layout)
win.setCentralWidget(central)


# ===== UPDATE FUNCTION =====
def update():
    global ppg_buffer, hr_buffer, last_gui, last_sample_time, last_peaks, peak_labels

    sample, ts = inlet.pull_sample(timeout=0.5)
    if sample:
        last_sample_time = time.time()
        ppg_buffer = np.roll(ppg_buffer, -1, axis=1)
        ppg_buffer[:, -1] = sample[:channels]

        # Normalize for display
        display_signal = ppg_buffer[0] - np.mean(ppg_buffer[0])
        display_signal = display_signal / (np.std(display_signal) + 1e-6)
        ppg_curve.setData(display_signal)

    # Stream timeout
    if time.time() - last_sample_time > STREAM_TIMEOUT:
        print(f"❌ No PPG data for {STREAM_TIMEOUT}s. Closing...")
        logout()
        return

    # HR/HRV every second
    if time.time() - last_gui >= GUI_SEC:
        hr, hrv, peaks = compute_hr_hrv(ppg_buffer, fs)
        stress_val = stress_index(hr, hrv)

        # Update labels
        hr_label.setText(f"HR: {hr:.1f} bpm")
        hrv_label.setText(f"HRV: {hrv:.3f} s")
        stress_label.setText(f"Stress: {stress_val:.1f}")

        # Clear old peak labels
        for label in peak_labels:
            ppg_plot.removeItem(label)
        peak_labels.clear()

        # Update peaks with R-R interval labels
        if len(peaks) > 1:
            display_signal = ppg_buffer[0] - np.mean(ppg_buffer[0])
            display_signal = display_signal / (np.std(display_signal) + 1e-6)

            peak_y = display_signal[peaks]
            peak_scatter.setData(peaks, peak_y)

            # Add R-R interval labels between peaks
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                rr_ms = (p2 - p1) / fs * 1000  # milliseconds
                inst_bpm = 60000 / rr_ms  # instantaneous BPM

                # Position label between peaks
                mid_x = (p1 + p2) / 2
                mid_y = max(display_signal[p1], display_signal[p2]) + 0.5

                # Create text label
                text = pg.TextItem(
                    f"{rr_ms:.0f} ms\n{inst_bpm:.0f} BPM",
                    color='#D32F2F',
                    anchor=(0.5, 0),
                    fill=pg.mkBrush(255, 255, 255, 180)
                )
                text.setFont(pg.QtGui.QFont("Arial", 9, pg.QtGui.QFont.Weight.Bold))
                text.setPos(mid_x, mid_y)
                ppg_plot.addItem(text)
                peak_labels.append(text)

                # Mark R peaks with label
                r_label = pg.TextItem('R', color='#B71C1C', anchor=(0.5, 1))
                r_label.setFont(pg.QtGui.QFont("Arial", 10, pg.QtGui.QFont.Weight.Bold))
                r_label.setPos(p2, display_signal[p2] - 0.3)
                ppg_plot.addItem(r_label)
                peak_labels.append(r_label)

        # Heartbeat beep
        new_peaks = [p for p in peaks if p not in last_peaks]
        if beep_obj and new_peaks:
            beep_obj.play()
            for _ in new_peaks:
                hr_buffer[-1] = hr

        last_peaks = peaks.copy()

        # Rolling HR buffer
        hr_buffer = np.roll(hr_buffer, -1)
        hr_buffer[-1] = hr
        hr_curve.setData(hr_buffer)
        hr_peak_scatter.setData([len(hr_buffer) - 1] * len(new_peaks), [hr] * len(new_peaks))

        # ===== SAVE DATA =====
        timestamp = time.time()
        with open(dataset_file, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, hr, hrv, stress_val])
        row = [timestamp] + list(sample[:channels]) + [hr]
        with open(raw_ppg_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

        hr_history.append(hr)
        hrv_history.append(hrv)
        last_gui = time.time()


# ===== LOGOUT FUNCTION =====
def logout():
    timer.stop()
    proc.terminate()
    print("🛑 Recording Stopped")
    app.quit()


logout_btn.clicked.connect(logout)

# ===== TIMER =====
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

print("🚀 PPG + HR Recording Started with ECG-style visualization")
win.show()
app.exec()