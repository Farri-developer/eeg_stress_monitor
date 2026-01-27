"""
Complete EEG + PPG + BP Biosignal Recorder - FINAL VERSION WITH STRESS LABELS
==============================================================================
Features:
- Real-time EEG, PPG, Stress graphs (all smooth)
- BP measurements with deltas
- Automatic dataset generation with Stress Labels (0, 1, 2)
- User name in folder structure
"""

import sys, os, csv, time, asyncio, subprocess, traceback
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from pylsl import StreamInlet, resolve_byprop
from bleak import BleakClient
from scipy.signal import butter, filtfilt, welch, find_peaks

# ================= CONFIGURATION =================
BASE_DIR = r"D:\DataSet"
os.makedirs(BASE_DIR, exist_ok=True)

# BP Device Configuration
BP_ADDRESS = "18:7A:93:12:26:AE"
BP_UUID = "00002a35-0000-1000-8000-00805f9b34fb"

# Signal Processing Configuration
EEG_FS = 256  # Muse EEG sampling rate
PPG_FS = 64   # Muse PPG sampling rate
DISPLAY_SEC = 10
STREAM_TIMEOUT = 30

# Smoothing parameters
EEG_SMOOTH_WINDOW = 15
PPG_SMOOTH_WINDOW = 5
STRESS_SMOOTH_WINDOW = 10

# ================= BP DECODER =================
def decode_bp(data):
    """Decode Bluetooth BP measurement data"""
    flags = data[0]
    systolic = int.from_bytes(data[1:3], "little")
    diastolic = int.from_bytes(data[3:5], "little")
    mean_art = int.from_bytes(data[5:7], "little")
    idx = 7
    if flags & 0x02:
        idx += 7
    pulse = int.from_bytes(data[idx:idx+2], "little") if flags & 0x04 else None
    if mean_art == 0:
        mean_art = round(diastolic + (systolic - diastolic)/3, 1)
    return systolic, diastolic, mean_art, pulse

# ================= SIGNAL PROCESSING =================
def bandpass_filter(data, low, high, fs, order=4):
    """Apply bandpass filter to signal"""
    try:
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data, axis=0)
    except:
        return data

def compute_band_powers(signal, fs):
    """Compute EEG frequency band powers"""
    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50)
    }

    powers = {}
    try:
        f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))

        for band_name, (low, high) in bands.items():
            mask = (f >= low) & (f < high)
            powers[band_name] = np.trapezoid(pxx[mask], f[mask])
    except:
        for band_name in bands.keys():
            powers[band_name] = 0.0

    return powers

def extract_ppg_hrv(ppg_signal, fs):
    """Extract HR and HRV metrics from PPG signal"""
    try:
        if len(ppg_signal) < fs * 2:
            return 0, 0, 0, 0

        peaks, _ = find_peaks(ppg_signal, distance=fs * 0.5)
        if len(peaks) < 2:
            return 0, 0, 0, 0

        rr_intervals = np.diff(peaks) / fs * 1000  # ms
        hr = 60 / (np.mean(rr_intervals) / 1000)  # bpm
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100

        return hr, sdnn, rmssd, pnn50
    except:
        return 0, 0, 0, 0

def compute_ppg_hr_hrv(ppg_signal, fs):
    """Extract HR and HRV from PPG signal for display"""
    try:
        if len(ppg_signal) < fs * 2:
            return 0, 0, []

        # Normalize signal
        ppg = ppg_signal - np.mean(ppg_signal)
        std = np.std(ppg)
        if std > 1e-6:
            ppg = ppg / std

        # Find peaks
        peaks, _ = find_peaks(ppg, distance=fs * 0.4, prominence=0.3)

        if len(peaks) < 2:
            return 0, 0, peaks

        # Calculate HR and HRV
        rr_intervals = np.diff(peaks) / fs
        hr = 60 / np.mean(rr_intervals)
        hrv = np.std(rr_intervals) * 1000  # Convert to ms

        return hr, hrv, peaks
    except:
        return 0, 0, []

def smooth_signal(signal, window=5):
    """Apply moving average smoothing"""
    try:
        if len(signal) < window:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')
    except:
        return signal

# ================= BP WORKER THREAD =================
class BPWorker(QtCore.QThread):
    """Thread for handling Bluetooth BP measurements"""
    reading = QtCore.Signal(int, int, float, object)
    status = QtCore.Signal(str)

    async def task(self):
        try:
            self.status.emit("🩺 BP: Connecting... Press device button")
            async with BleakClient(BP_ADDRESS, timeout=12) as client:
                got_reading = False

                def handler(_, data):
                    nonlocal got_reading
                    if not got_reading:
                        got_reading = True
                        self.reading.emit(*decode_bp(data))

                await client.start_notify(BP_UUID, handler)

                for _ in range(45):
                    if got_reading:
                        break
                    await asyncio.sleep(1)

                await client.stop_notify(BP_UUID)
                self.status.emit("BP: Measurement complete ✓")

        except Exception as e:
            self.status.emit(f"BP Error: {str(e)}")

    def run(self):
        asyncio.run(self.task())

# ================= MAIN GUI APPLICATION =================
class BiosignalRecorder(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Biosignal Recorder - EEG + PPG + BP + Stress Labels")
        self.setGeometry(100, 100, 1600, 900)

        # ===== GET USER INFO =====
        self._get_user_info()

        # ===== SESSION SETUP =====
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(BASE_DIR, f"{self.user_name}({self.user_id})_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)

        # ===== FILE PATHS =====
        self.eeg_file = os.path.join(self.folder, "eeg_raw.csv")
        self.ppg_file = os.path.join(self.folder, "ppg_raw.csv")
        self.bp_file = os.path.join(self.folder, "bp_measurements.csv")
        self.dataset_file = os.path.join(self.folder, "dataset_features.csv")

        # Initialize CSV files
        self._init_csv_files()

        # ===== STATE VARIABLES =====
        self.proc = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.recording = False
        self.streaming = False
        self.base_bp = None
        self.question_end_bp = None
        self.last_sample_time = time.time()

        self.eeg_channels = 4
        self.ppg_channels = 4

        # ===== DATA BUFFERS =====
        self.eeg_buffer = np.zeros((4, EEG_FS * DISPLAY_SEC))
        self.ppg_buffer = None
        self.hr_history = []
        self.stress_history = []

        # ===== SETUP UI =====
        self._setup_ui()

        # ===== START TIMERS =====
        self.data_timer = QtCore.QTimer()
        self.data_timer.timeout.connect(self._update_data)
        self.data_timer.start(20)

        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self._update_graphs)
        self.graph_timer.start(50)

    def _get_user_info(self):
        """Get user name and ID"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("User Information")
        dialog.setModal(True)

        layout = QtWidgets.QFormLayout()

        name_input = QtWidgets.QLineEdit()
        name_input.setPlaceholderText("e.g., Farhan")

        id_input = QtWidgets.QLineEdit()
        id_input.setPlaceholderText("e.g., 22-arid-3981")

        layout.addRow("Name:", name_input)
        layout.addRow("ID:", id_input)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)

        layout.addRow(btn_box)
        dialog.setLayout(layout)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.user_name = name_input.text().strip() or "User"
            self.user_id = id_input.text().strip() or "Unknown"
        else:
            self.user_name = "User"
            self.user_id = "Unknown"

    def _init_csv_files(self):
        """Initialize all CSV files with headers"""
        with open(self.eeg_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "EEG1", "EEG2", "EEG3", "EEG4"])

        with open(self.ppg_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "PPG1", "PPG2", "PPG3", "PPG4", "HR", "HRV"])

        with open(self.bp_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["time", "label", "SYS", "DIA", "MAP", "PULSE",
                                   "DeltaSYS", "DeltaDIA", "DeltaPulse"])

    def _setup_ui(self):
        """Create the user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # ===== TOP: SESSION INFO =====
        info_layout = QtWidgets.QHBoxLayout()
        session_label = QtWidgets.QLabel(f"📁 {self.user_name} ({self.user_id})")
        session_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        folder_label = QtWidgets.QLabel(f"Folder: {self.folder}")
        folder_label.setStyleSheet("font-size: 11px; color: #666;")
        info_layout.addWidget(session_label)
        info_layout.addStretch()
        info_layout.addWidget(folder_label)
        main_layout.addLayout(info_layout)

        # ===== MIDDLE: GRAPHS =====
        graphs_layout = QtWidgets.QHBoxLayout()

        # EEG Column
        eeg_container = QtWidgets.QVBoxLayout()
        eeg_label = QtWidgets.QLabel("🧠 EEG Signals (Smooth)")
        eeg_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        eeg_container.addWidget(eeg_label)

        self.eeg_plot = pg.PlotWidget(title="EEG Activity")
        self.eeg_plot.setBackground('#1a1a1a')
        self.eeg_plot.showGrid(x=True, y=True, alpha=0.3)

        colors = ['#00FF00', '#FF00FF', '#00FFFF', '#FFFF00']
        self.eeg_curves = []
        for i, color in enumerate(colors):
            curve = self.eeg_plot.plot(pen=pg.mkPen(color, width=2), name=f'Ch{i+1}')
            self.eeg_curves.append(curve)

        self.eeg_plot.addLegend()
        eeg_container.addWidget(self.eeg_plot)

        # Band Powers
        self.band_labels = {}
        band_layout = QtWidgets.QGridLayout()
        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        colors_band = ["#FF5252", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]

        for i, (band, color) in enumerate(zip(bands, colors_band)):
            label = QtWidgets.QLabel(f"{band}: 0")
            label.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold; padding: 5px;")
            self.band_labels[band] = label
            band_layout.addWidget(label, 0, i)

        eeg_container.addLayout(band_layout)
        graphs_layout.addLayout(eeg_container)

        # PPG Column
        ppg_container = QtWidgets.QVBoxLayout()
        ppg_label = QtWidgets.QLabel("❤️ PPG Signal (Smooth)")
        ppg_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        ppg_container.addWidget(ppg_label)

        self.ppg_plot = pg.PlotWidget(title="PPG Waveform")
        self.ppg_plot.setBackground('#1a1a1a')
        self.ppg_plot.showGrid(x=True, y=True, alpha=0.3)

        self.ppg_curve = self.ppg_plot.plot(pen=pg.mkPen('#E53935', width=2.5))
        self.ppg_peaks = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 180))
        self.ppg_plot.addItem(self.ppg_peaks)
        ppg_container.addWidget(self.ppg_plot)

        # Metrics
        metrics_layout = QtWidgets.QHBoxLayout()
        self.hr_label = QtWidgets.QLabel("HR: -- bpm")
        self.hr_label.setStyleSheet("font-size: 16px; color: #00FF00; font-weight: bold; "
                                     "background: #1a1a1a; padding: 8px; border-radius: 5px;")
        self.hrv_label = QtWidgets.QLabel("HRV: -- ms")
        self.hrv_label.setStyleSheet("font-size: 16px; color: #FF4500; font-weight: bold; "
                                      "background: #1a1a1a; padding: 8px; border-radius: 5px;")
        metrics_layout.addWidget(self.hr_label)
        metrics_layout.addWidget(self.hrv_label)
        ppg_container.addLayout(metrics_layout)
        graphs_layout.addLayout(ppg_container)

        # Stress Column
        stress_container = QtWidgets.QVBoxLayout()
        stress_label = QtWidgets.QLabel("📊 Stress Index (Smooth)")
        stress_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        stress_container.addWidget(stress_label)

        self.stress_plot = pg.PlotWidget(title="Stress Over Time")
        self.stress_plot.setBackground('#1a1a1a')
        self.stress_plot.showGrid(x=True, y=True, alpha=0.3)
        self.stress_curve = self.stress_plot.plot(pen=pg.mkPen('#FFD700', width=2.5))
        stress_container.addWidget(self.stress_plot)

        self.stress_label = QtWidgets.QLabel("Stress: --")
        self.stress_label.setStyleSheet("font-size: 16px; color: #FFD700; font-weight: bold; "
                                         "background: #1a1a1a; padding: 8px; border-radius: 5px;")
        stress_container.addWidget(self.stress_label)
        graphs_layout.addLayout(stress_container)

        main_layout.addLayout(graphs_layout)

        # ===== BOTTOM: STATUS AND CONTROLS =====
        self.status_label = QtWidgets.QLabel("Status: Ready to start")
        self.status_label.setStyleSheet("font-size: 13px; color: #FFF; background: #424242; "
                                        "padding: 8px; border-radius: 5px;")
        self.bp_status_label = QtWidgets.QLabel("BP: Not measured")
        self.bp_status_label.setStyleSheet("font-size: 13px; color: #FFF; background: #424242; "
                                           "padding: 8px; border-radius: 5px;")
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.bp_status_label)

        # Buttons
        button_layout = QtWidgets.QGridLayout()

        self.btn_baseline_bp = self._create_button("🩺 1. Baseline BP", "#2196F3", True)
        self.btn_stream_start = self._create_button("▶ 2. Start Stream", "#4CAF50", True)
        button_layout.addWidget(self.btn_baseline_bp, 0, 0)
        button_layout.addWidget(self.btn_stream_start, 0, 1)

        self.btn_start_recording = self._create_button("⏺ 3. Start Recording", "#E91E63", False)
        self.btn_stop_recording = self._create_button("⏹ 4. Stop Recording", "#9C27B0", False)
        button_layout.addWidget(self.btn_start_recording, 1, 0)
        button_layout.addWidget(self.btn_stop_recording, 1, 1)

        self.btn_stream_stop = self._create_button("⏹ 5. Stop Stream", "#F44336", False)
        self.btn_question_end_bp = self._create_button("💓 6. Question-End BP", "#FF9800", False)
        button_layout.addWidget(self.btn_stream_stop, 2, 0)
        button_layout.addWidget(self.btn_question_end_bp, 2, 1)

        self.btn_generate_dataset = self._create_button("📊 7. Generate Dataset with Labels", "#00BCD4", False)
        self.btn_exit = self._create_button("🚪 Exit", "#607D8B", True)
        button_layout.addWidget(self.btn_generate_dataset, 3, 0)
        button_layout.addWidget(self.btn_exit, 3, 1)

        main_layout.addLayout(button_layout)

        # Connect signals
        self.btn_baseline_bp.clicked.connect(self.measure_baseline_bp)
        self.btn_stream_start.clicked.connect(self.start_stream)
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        self.btn_stream_stop.clicked.connect(self.stop_stream)
        self.btn_question_end_bp.clicked.connect(self.measure_question_end_bp)
        self.btn_generate_dataset.clicked.connect(self.generate_dataset)
        self.btn_exit.clicked.connect(self.exit_application)

    def _create_button(self, text, color, enabled=True):
        btn = QtWidgets.QPushButton(text)
        btn.setEnabled(enabled)
        btn.setMinimumHeight(45)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: {color}DD; }}
            QPushButton:disabled {{ background-color: #555; color: #999; }}
        """)
        return btn

    # ================= BP MEASUREMENTS =================
    def measure_baseline_bp(self):
        self.status_label.setText("🩺 Measuring baseline BP...")
        self.bp_thread = BPWorker()
        self.bp_thread.reading.connect(self._save_baseline_bp)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def _save_baseline_bp(self, sys, dia, map_val, pulse):
        self.base_bp = (sys, dia, pulse)
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, "Baseline", sys, dia, map_val, pulse, 0, 0, 0])
        self.bp_status_label.setText(f"✓ Baseline: {sys}/{dia} mmHg, P {pulse}")
        self.status_label.setText("✅ Baseline saved")

    def measure_question_end_bp(self):
        if not self.base_bp:
            QtWidgets.QMessageBox.warning(self, "Error", "Measure Baseline BP first!")
            return
        self.status_label.setText("💓 Measuring Question-End BP...")
        self.bp_thread = BPWorker()
        self.bp_thread.reading.connect(self._save_question_end_bp)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def _save_question_end_bp(self, sys, dia, map_val, pulse):
        delta_sys = sys - self.base_bp[0]
        delta_dia = dia - self.base_bp[1]
        delta_pulse = pulse - self.base_bp[2]
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, "Question-End", sys, dia, map_val, pulse,
                                   delta_sys, delta_dia, delta_pulse])
        self.question_end_bp = (sys, dia, pulse, delta_sys, delta_dia, delta_pulse)
        self.bp_status_label.setText(
            f"✓ Q-End: {sys}/{dia}, P {pulse} (Δ:{delta_sys:+d}/{delta_dia:+d}/{delta_pulse:+d})"
        )
        self.status_label.setText("✅ Question-End BP saved")
        self.btn_generate_dataset.setEnabled(True)

    # ================= STREAM MANAGEMENT =================
    def start_stream(self):
        if self.streaming:
            return
        try:
            self.status_label.setText("🔌 Connecting...")
            QtWidgets.QApplication.processEvents()

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "muselsl", "stream", "--ppg"],
                creationflags=creationflags
            )
            time.sleep(6)

            eeg_streams = resolve_byprop("type", "EEG", timeout=10)
            ppg_streams = resolve_byprop("type", "PPG", timeout=10)

            if not eeg_streams or not ppg_streams:
                raise RuntimeError("Streams not found")

            self.eeg_inlet = StreamInlet(eeg_streams[0])
            self.ppg_inlet = StreamInlet(ppg_streams[0])
            self.ppg_channels = self.ppg_inlet.info().channel_count()
            self.ppg_buffer = np.zeros((self.ppg_channels, PPG_FS * DISPLAY_SEC))

            self.streaming = True
            self.last_sample_time = time.time()

            self.status_label.setText("✅ Stream connected - Graphs live")
            self.btn_stream_start.setEnabled(False)
            self.btn_stream_stop.setEnabled(True)
            self.btn_start_recording.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            if self.proc:
                self.proc.terminate()
                self.proc = None

    def stop_stream(self):
        if self.recording:
            self.stop_recording()
        if self.proc:
            try:
                if sys.platform == 'win32':
                    import signal
                    os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.terminate()
            except:
                pass
            self.proc = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.streaming = False
        self.status_label.setText("⏹ Stream stopped")
        self.btn_stream_start.setEnabled(True)
        self.btn_stream_stop.setEnabled(False)
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(False)
        self.btn_question_end_bp.setEnabled(True)

    def start_recording(self):
        if not self.streaming:
            return
        self.recording = True
        self.status_label.setText("⏺ Recording...")
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(True)

    def stop_recording(self):
        self.recording = False
        self.status_label.setText("⏹ Stopped")
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)

    # ================= DATA UPDATE =================
    def _update_data(self):
        if not self.streaming:
            return
        try:
            if self.eeg_inlet:
                eeg_sample, eeg_ts = self.eeg_inlet.pull_sample(timeout=0.0)
                if eeg_sample:
                    self.last_sample_time = time.time()
                    self.eeg_buffer = np.roll(self.eeg_buffer, -1, axis=1)
                    self.eeg_buffer[:, -1] = eeg_sample[:4]
                    if self.recording:
                        with open(self.eeg_file, "a", newline="") as f:
                            csv.writer(f).writerow([eeg_ts] + eeg_sample[:4])

            if self.ppg_inlet and self.ppg_buffer is not None:
                ppg_sample, ppg_ts = self.ppg_inlet.pull_sample(timeout=0.0)
                if ppg_sample:
                    self.last_sample_time = time.time()
                    actual_channels = min(len(ppg_sample), self.ppg_channels)
                    self.ppg_buffer = np.roll(self.ppg_buffer, -1, axis=1)
                    self.ppg_buffer[:actual_channels, -1] = ppg_sample[:actual_channels]

                    hr, hrv, peaks = compute_ppg_hr_hrv(self.ppg_buffer[0], PPG_FS)

                    if self.recording:
                        ppg_data = list(ppg_sample[:actual_channels])
                        while len(ppg_data) < 4:
                            ppg_data.append(0)
                        with open(self.ppg_file, "a", newline="") as f:
                            csv.writer(f).writerow([ppg_ts] + ppg_data[:4] + [hr, hrv])
        except Exception as e:
            print(f"Data error: {e}")

    def _update_graphs(self):
        if not self.streaming:
            return
        try:
            # EEG - SMOOTH
            for i, curve in enumerate(self.eeg_curves):
                signal = self.eeg_buffer[i]
                smoothed = smooth_signal(signal, window=EEG_SMOOTH_WINDOW)
                offset_signal = smoothed + (i * 30)
                curve.setData(offset_signal)

            # Band powers
            if np.any(self.eeg_buffer):
                all_powers = {"Delta": 0, "Theta": 0, "Alpha": 0, "Beta": 0, "Gamma": 0}
                for ch in range(4):
                    powers = compute_band_powers(self.eeg_buffer[ch], EEG_FS)
                    for band in all_powers:
                        all_powers[band] += powers[band]
                for band in all_powers:
                    all_powers[band] /= 4
                    self.band_labels[band].setText(f"{band}: {all_powers[band]:.0f}")

            # PPG - SMOOTH
            if self.ppg_buffer is not None and np.any(self.ppg_buffer):
                ppg_signal = self.ppg_buffer[0]
                ppg_mean = np.mean(ppg_signal)
                ppg_std = np.std(ppg_signal)
                if ppg_std > 1e-6:
                    ppg_norm = (ppg_signal - ppg_mean) / ppg_std
                else:
                    ppg_norm = ppg_signal - ppg_mean

                ppg_smooth = smooth_signal(ppg_norm, window=PPG_SMOOTH_WINDOW)
                self.ppg_curve.setData(ppg_smooth)

                hr, hrv, peaks = compute_ppg_hr_hrv(ppg_signal, PPG_FS)

                if hr > 0:
                    self.hr_label.setText(f"HR: {hr:.1f} bpm")
                    self.hrv_label.setText(f"HRV: {hrv:.1f} ms")

                    self.hr_history.append(hr)
                    if len(self.hr_history) > 100:
                        self.hr_history.pop(0)

                    # Stress index
                    if np.any(self.eeg_buffer):
                        powers = compute_band_powers(self.eeg_buffer[0], EEG_FS)
                        beta_alpha = powers["Beta"] / (powers["Alpha"] + 1e-6)
                        hr_hrv_ratio = hr / (hrv + 1e-6)
                        stress = (beta_alpha + hr_hrv_ratio) / 2

                        self.stress_label.setText(f"Stress: {stress:.2f}")
                        self.stress_history.append(stress)
                        if len(self.stress_history) > 100:
                            self.stress_history.pop(0)

                        # SMOOTH stress graph
                        stress_smooth = smooth_signal(np.array(self.stress_history),
                                                     window=STRESS_SMOOTH_WINDOW)
                        self.stress_curve.setData(stress_smooth)

                    if len(peaks) > 0:
                        peak_y = ppg_smooth[peaks]
                        self.ppg_peaks.setData(peaks, peak_y)
        except Exception as e:
            print(f"Graph error: {e}")

    # ================= DATASET GENERATION WITH LABELS =================
    def generate_dataset(self):
        try:
            import pandas as pd

            self.status_label.setText("📊 Generating dataset with stress labels...")
            QtWidgets.QApplication.processEvents()

            # Load data
            eeg_df = pd.read_csv(self.eeg_file)
            ppg_df = pd.read_csv(self.ppg_file)
            bp_df = pd.read_csv(self.bp_file)

            if len(eeg_df) == 0:
                raise ValueError("No EEG data!")

            # BP features
            baseline = bp_df[bp_df['label'] == 'Baseline'].iloc[0] if len(bp_df) > 0 else None
            question_end = bp_df[bp_df['label'] == 'Question-End'].iloc[-1] if len(bp_df[bp_df['label'] == 'Question-End']) > 0 else None

            if baseline is not None and question_end is not None:
                bp_features = {
                    'DeltaSYS': question_end['DeltaSYS'],
                    'DeltaDIA': question_end['DeltaDIA'],
                    'DeltaPULSE': question_end['DeltaPulse']
                }
            else:
                bp_features = {'DeltaSYS': 0, 'DeltaDIA': 0, 'DeltaPULSE': 0}

            # Process 5-second windows
            WIN_SEC = 5
            WIN_SAMPLES = EEG_FS * WIN_SEC
            rows = []

            eeg_data = eeg_df[['EEG1', 'EEG2', 'EEG3', 'EEG4']].values
            num_windows = len(eeg_data) // WIN_SAMPLES

            for w in range(num_windows):
                s = w * WIN_SAMPLES
                e = s + WIN_SAMPLES
                if e > len(eeg_data):
                    break

                eeg_win = eeg_data[s:e]
                eeg_win_filtered = bandpass_filter(eeg_win, 0.5, 45, EEG_FS)

                features = {'Window': w}

                # EEG features
                for ch in range(4):
                    powers = compute_band_powers(eeg_win_filtered[:, ch], EEG_FS)
                    for band, power in powers.items():
                        features[f'EEG{ch+1}_{band}'] = power
                    # Beta/Alpha ratio
                    features[f'EEG{ch+1}_BetaAlpha'] = powers["Beta"] / (powers["Alpha"] + 1e-6)

                # PPG features
                ppg_start_idx = int(s * PPG_FS / EEG_FS)
                ppg_end_idx = int(e * PPG_FS / EEG_FS)

                if ppg_end_idx < len(ppg_df):
                    ppg_win = ppg_df.iloc[ppg_start_idx:ppg_end_idx]
                    ppg_cols = [c for c in ['PPG1', 'PPG2', 'PPG3', 'PPG4'] if c in ppg_win.columns]

                    if len(ppg_cols) > 0:
                        ppg_data_vals = ppg_win[ppg_cols].values
                        features['PPG_Mean'] = np.mean(ppg_data_vals)
                        features['PPG_STD'] = np.std(ppg_data_vals)

                        # HRV metrics from PPG channels
                        hr_all, sdnn_all, rmssd_all, pnn50_all = [], [], [], []
                        for ch_idx in range(len(ppg_cols)):
                            ppg_ch_data = ppg_data_vals[:, ch_idx]
                            hr, sdnn, rmssd, pnn50 = extract_ppg_hrv(ppg_ch_data, PPG_FS)
                            hr_all.append(hr)
                            sdnn_all.append(sdnn)
                            rmssd_all.append(rmssd)
                            pnn50_all.append(pnn50)

                        features['PPG_HR'] = np.mean(hr_all)
                        features['PPG_SDNN'] = np.mean(sdnn_all)
                        features['PPG_RMSSD'] = np.mean(rmssd_all)
                        features['PPG_pNN50'] = np.mean(pnn50_all)
                    else:
                        features['PPG_Mean'] = 0
                        features['PPG_STD'] = 0
                        features['PPG_HR'] = 0
                        features['PPG_SDNN'] = 0
                        features['PPG_RMSSD'] = 0
                        features['PPG_pNN50'] = 0
                else:
                    features['PPG_Mean'] = 0
                    features['PPG_STD'] = 0
                    features['PPG_HR'] = 0
                    features['PPG_SDNN'] = 0
                    features['PPG_RMSSD'] = 0
                    features['PPG_pNN50'] = 0

                # BP features
                features.update(bp_features)
                rows.append(features)

            # Create DataFrame
            final_df = pd.DataFrame(rows)

            # ===== CALCULATE STRESS SCORE =====
            def calculate_stress_score(row):
                beta_alpha_cols = [f'EEG{i}_BetaAlpha' for i in range(1, 5)]
                mean_betaalpha = np.mean([row[col] for col in beta_alpha_cols])
                score = row['PPG_HR'] + (1 / (row['PPG_RMSSD'] + 1e-6)) + mean_betaalpha
                return score

            final_df['StressScore'] = final_df.apply(calculate_stress_score, axis=1)

            # ===== ASSIGN STRESS LABELS (0, 1, 2) =====
            final_df['Stress_Label'] = pd.qcut(
                final_df['StressScore'],
                q=3,
                labels=['0', '1', '2'],
                duplicates='drop'
            )

            # Save
            final_df.to_csv(self.dataset_file, index=False)

            self.status_label.setText(f"✅ Dataset: {len(rows)} windows with labels")

            # Count labels
            label_counts = final_df['Stress_Label'].value_counts().to_dict()
            label_str = ", ".join([f"Label {k}: {v}" for k, v in sorted(label_counts.items())])

            QtWidgets.QMessageBox.information(
                self, "Success",
                f"Dataset generated!\n\n"
                f"Windows: {len(rows)}\n"
                f"Stress Labels: {label_str}\n\n"
                f"Location:\n{self.folder}\n\n"
                f"Files:\n"
                f"• eeg_raw.csv\n"
                f"• ppg_raw.csv\n"
                f"• bp_measurements.csv\n"
                f"• dataset_features.csv (with Stress_Label column)"
            )

        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def exit_application(self):
        if self.streaming:
            self.stop_stream()
        QtWidgets.QApplication.quit()

    def closeEvent(self, event):
        if self.streaming:
            self.stop_stream()
        event.accept()

# ================= MAIN =================
if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "--break-system-packages"])
        import pandas as pd

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark theme
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.GlobalColor.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.GlobalColor.black)
    app.setPalette(palette)

    window = BiosignalRecorder()
    window.show()

    sys.exit(app.exec())