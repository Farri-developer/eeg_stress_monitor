
import sys, os, csv, time, asyncio, subprocess, traceback
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from pylsl import StreamInlet, resolve_byprop
from bleak import BleakClient
from scipy.signal import butter, filtfilt, welch, find_peaks

# ==================== CONFIGURATION ====================
BASE_DIR = r"D:\DataSet"
os.makedirs(BASE_DIR, exist_ok=True)

# BP Device Configuration
BP_ADDRESS = "18:7A:93:12:26:AE"
BP_UUID = "00002a35-0000-1000-8000-00805f9b34fb"

# Signal Processing Configuration
EEG_FS_EXPECTED = 256  # Expected Muse EEG sampling rate
PPG_FS_EXPECTED = 64   # Expected Muse PPG sampling rate
DISPLAY_SEC = 10
STREAM_TIMEOUT = 30

# Smoothing parameters
EEG_SMOOTH_WINDOW = 15
PPG_SMOOTH_WINDOW = 5
STRESS_SMOOTH_WINDOW = 10

# Research Configuration - TLX-Based Labeling (UNCHANGED)
TLX_THRESHOLDS = {
    'LOW_MAX': 1.5,      # 0 – 1.5  → Low
    'MEDIUM_MAX': 3.0,   # 1.6 – 3.0 → Medium
    # 3.1 – 4.0 → High
}


# Validation thresholds
SAMPLING_RATE_TOLERANCE = 0.05  # 5% tolerance
MIN_GAMMA_FS = 100  # Minimum sampling rate for Gamma band


# ==================== HELPER FUNCTIONS ====================
def decode_bp(data):
    """Decode Bluetooth BP measurement data"""
    flags = data[0]
    systolic = int.from_bytes(data[1:3], "little")
    diastolic = int.from_bytes(data[3:5], "little")
    mean_art = int.from_bytes(data[5:7], "little")
    idx = 7
    if flags & 0x02:
        idx += 7
    pulse = int.from_bytes(data[idx:idx + 2], "little") if flags & 0x04 else None
    if mean_art == 0:
        mean_art = round(diastolic + (systolic - diastolic) / 3, 1)
    return systolic, diastolic, mean_art, pulse


def bandpass_filter(data, low, high, fs, order=4):
    """Apply bandpass filter to signal"""
    try:
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data, axis=0)
    except:
        return data


def compute_band_powers(signal, fs, include_gamma=True):
    """
    Compute EEG frequency band powers with Gamma safety check

    Args:
        signal: EEG signal array
        fs: Actual sampling rate
        include_gamma: Whether to include Gamma band (requires fs >= 100 Hz)

    Returns:
        Dictionary of band powers
    """
    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
    }

    # Only include Gamma if sampling rate is sufficient
    if include_gamma and fs >= MIN_GAMMA_FS:
        bands["Gamma"] = (30, 50)

    powers = {}
    try:
        f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
        for band_name, (low, high) in bands.items():
            mask = (f >= low) & (f < high)
            powers[band_name] = np.trapezoid(pxx[mask], f[mask])
    except:
        for band_name in bands.keys():
            powers[band_name] = 0.0

    # Add Gamma = 0 if not included
    if "Gamma" not in powers:
        powers["Gamma"] = 0.0

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
        ppg = ppg_signal - np.mean(ppg_signal)
        std = np.std(ppg)
        if std > 1e-6:
            ppg = ppg / std
        peaks, _ = find_peaks(ppg, distance=fs * 0.4, prominence=0.3)
        if len(peaks) < 2:
            return 0, 0, peaks
        rr_intervals = np.diff(peaks) / fs
        hr = 60 / np.mean(rr_intervals)
        hrv = np.std(rr_intervals) * 1000
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


def tlx_to_stress_label(score):

    if score <= TLX_THRESHOLDS['LOW_MAX']:
        return 0
    elif score <= TLX_THRESHOLDS['MEDIUM_MAX']:
        return 1
    else:
        return 2


def validate_sampling_rate(actual_fs, expected_fs, signal_type):
    """
    Validate actual sampling rate against expected

    Returns:
        (is_valid, deviation_percent, warning_message)
    """
    deviation = abs(actual_fs - expected_fs) / expected_fs
    is_valid = deviation <= SAMPLING_RATE_TOLERANCE

    warning = None
    if not is_valid:
        warning = f"⚠️ {signal_type} sampling rate deviation: {deviation*100:.2f}% " \
                  f"(Expected: {expected_fs}Hz, Actual: {actual_fs:.2f}Hz)"

    return is_valid, deviation * 100, warning


def check_signal_quality(signal, signal_name, timestamps=None):
    """
    Check signal quality and detect issues

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for flat signal (zero variance)
    if np.std(signal) < 1e-6:
        warnings.append(f"⚠️ {signal_name}: Flat signal detected (zero variance)")

    # Check for constant values
    if len(np.unique(signal)) == 1:
        warnings.append(f"⚠️ {signal_name}: Constant signal (all values identical)")

    # Check for outliers (z-score > 5)
    try:
        z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-6))
        outlier_count = np.sum(z_scores > 5)
        if outlier_count > len(signal) * 0.01:  # > 1% outliers
            warnings.append(f"⚠️ {signal_name}: {outlier_count} outliers detected (z-score > 5)")
    except:
        pass

    # Check for missing timestamps
    if timestamps is not None and len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        if len(time_diffs) > 0:
            median_diff = np.median(time_diffs)
            gaps = np.sum(time_diffs > median_diff * 3)
            if gaps > 0:
                warnings.append(f"⚠️ {signal_name}: {gaps} timestamp gaps detected")

    return warnings


# ==================== BP WORKER THREAD ====================
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
                self.status.emit("✅ BP: Measurement complete")
        except Exception as e:
            self.status.emit(f"❌ BP Error: {str(e)}")

    def run(self):
        asyncio.run(self.task())


# ==================== NASA-TLX DIALOG ====================

class NASATLXDialog(QtWidgets.QDialog):
    """
    FINAL SELF-REPORT SCREEN (3-DIMENSION MODEL)
    5-Point Likert Scale (0–4)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coding Workload & Stress Assessment")
        self.setModal(True)
        self.resize(900, 650)
        self.setMinimumHeight(500)
        self.current_hr = 0
        self.current_hrv = 0

        self.setStyleSheet("QDialog { background: #1a1a1a; }")

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Scroll Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(20)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        title = QtWidgets.QLabel("🧠 Coding Workload & Stress Assessment")
        title.setStyleSheet("font-size:22px;font-weight:bold;color:white;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        instruction = QtWidgets.QLabel(
            "Please rate your experience during this coding task.\n"
            "There are no right or wrong answers."
        )
        instruction.setStyleSheet("font-size:14px;color:#CCC;")
        instruction.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(instruction)

        self.questions = {
            "Mental_Demand": "How mentally demanding was this coding task? (Did the task require a lot of thinking note, concentration, or problem-solving?)",
            "Effort": "How much effort did you put into completing this task? (How hard did you have to work to achieve your performance?)",
            "Frustration": "How frustrated or stressed did you feel?(Did you feel irritated, anxious, or discouraged while coding?)"
        }

        self.sliders = {}

        for key, question in self.questions.items():
            frame = QtWidgets.QFrame()
            frame.setStyleSheet("QFrame{background:#222;border-radius:10px;padding:15px;}")
            frame_layout = QtWidgets.QVBoxLayout(frame)

            label = QtWidgets.QLabel(question)
            label.setStyleSheet("font-size:16px;color:white;font-weight:bold;")
            label.setWordWrap(True)
            frame_layout.addWidget(label)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(4)
            slider.setValue(2)
            slider.setTickInterval(1)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            frame_layout.addWidget(slider)

            self.sliders[key] = slider
            layout.addWidget(frame)

        # ===== Buttons OUTSIDE Scroll =====
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.clicked.connect(self.reject)

        submit_btn = QtWidgets.QPushButton("Submit")
        submit_btn.setMinimumHeight(40)
        submit_btn.setStyleSheet("background:#4CAF50;color:white;font-weight:bold;")
        submit_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)

        main_layout.addLayout(button_layout)

    def get_scores(self):
        return {k: v.value() for k, v in self.sliders.items()}

    def get_overall_score(self):
        scores = self.get_scores()
        return sum(scores.values()) / 3



class BiosignalRecorder(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Research-Grade Stress Detection System - EEG + PPG + BP [CORRECTED]")
        self.setGeometry(50, 50, 1700, 950)
        self.current_hr = 0
        self.current_hrv = 0

        # Get user info
        self._get_user_info()

        # Setup session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(BASE_DIR, f"{self.user_name}({self.user_id})_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)

        # File paths
        self.eeg_file = os.path.join(self.folder, "eeg_raw.csv")
        self.ppg_file = os.path.join(self.folder, "ppg_raw.csv")
        self.bp_file = os.path.join(self.folder, "bp_measurements.csv")



        # Initialize CSV files
        self._init_csv_files()

        # State variables
        self.proc = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.recording = False
        self.streaming = False
        self.base_bp = None
        self.question_end_bp = None
        self.tlx_scores = None

        self.eeg_channels = 4
        self.ppg_channels = 4
        self.recording_start_time = None
        self.recording_duration = 0

        # ===== CORRECTED: Chunk-based buffers with LSL timestamps =====

        self.first_eeg_timestamp = None
        self.first_ppg_timestamp = None

        # Data buffers for display
        self.eeg_buffer = np.zeros((4, EEG_FS_EXPECTED * DISPLAY_SEC))
        self.ppg_buffer = None
        self.hr_history = []
        self.stress_history = []

        # Setup UI
        self._setup_ui()

        # Start timers
        self.data_timer = QtCore.QTimer()
        self.data_timer.timeout.connect(self._update_data)
        self.data_timer.start(20)

        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self._update_graphs)
        self.graph_timer.start(50)

        self.duration_timer = QtCore.QTimer()
        self.duration_timer.timeout.connect(self._update_duration)
        self.duration_timer.start(1000)

    def _get_user_info(self):
        """Get user name and ID"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("👤 User Information")
        dialog.setModal(True)
        dialog.setStyleSheet("""
            QDialog {
                background: #2a2a2a;
            }
            QLabel {
                color: #FFF;
                font-size: 13px;
            }
            QLineEdit {
                background: #1a1a1a;
                color: #FFF;
                border: 2px solid #444;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #2196F3;
            }
        """)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(15)

        title = QtWidgets.QLabel("Enter User Information")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        layout.addWidget(title)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        name_input = QtWidgets.QLineEdit()
        name_input.setPlaceholderText("e.g., Farhan")
        id_input = QtWidgets.QLineEdit()
        id_input.setPlaceholderText("e.g., 22-arid-3981")

        form_layout.addRow("Name:", name_input)
        form_layout.addRow("ID:", id_input)

        layout.addLayout(form_layout)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 70px;
            }
            QPushButton:hover {
                background: #1976D2;
            }
        """)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

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
            csv.writer(f).writerow(["lsl_timestamp", "EEG1", "EEG2", "EEG3", "EEG4"])
        with open(self.ppg_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["lsl_timestamp", "PPG1", "PPG2", "PPG3", "PPG4", "HR", "HRV"])
        with open(self.bp_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["time", "label", "SYS", "DIA", "MAP", "PULSE",
                                    "DeltaSYS", "DeltaDIA", "DeltaPulse"])

    def _setup_ui(self):
        """Create the user interface"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ===== SESSION INFO =====
        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a237e, stop:1 #0d47a1);
                border-radius: 10px;
                padding: 15px;
            }
        """)
        info_layout = QtWidgets.QHBoxLayout(info_frame)

        session_label = QtWidgets.QLabel(f"👤 {self.user_name} ({self.user_id})")
        session_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFF;")

        folder_label = QtWidgets.QLabel(f"📁 {self.folder}")
        folder_label.setStyleSheet("font-size: 11px; color: #BBB;")

        info_layout.addWidget(session_label)
        info_layout.addStretch()
        info_layout.addWidget(folder_label)

        main_layout.addWidget(info_frame)

        # ===== GRAPHS =====
        graphs_layout = QtWidgets.QHBoxLayout()
        graphs_layout.setSpacing(10)

        # EEG Column
        eeg_container = self._create_graph_container(
            "🧠 EEG Signals (4 Channels)",
            "EEG Activity"
        )
        self.eeg_plot = eeg_container['plot']
        colors = ['#00FF00', '#FF00FF', '#00FFFF', '#FFFF00']
        self.eeg_curves = []
        for i, color in enumerate(colors):
            curve = self.eeg_plot.plot(pen=pg.mkPen(color, width=2.5), name=f'Ch{i + 1}')
            self.eeg_curves.append(curve)
        self.eeg_plot.addLegend()

        # Band Powers
        self.band_labels = {}
        band_layout = QtWidgets.QGridLayout()
        band_layout.setSpacing(5)
        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        colors_band = ["#FF5252", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
        for i, (band, color) in enumerate(zip(bands, colors_band)):
            label = QtWidgets.QLabel(f"{band}: 0.00")
            label.setStyleSheet(f"""
                font-size: 12px; 
                color: {color}; 
                font-weight: bold; 
                padding: 6px;
                background: #1a1a1a;
                border-radius: 5px;
            """)
            self.band_labels[band] = label
            band_layout.addWidget(label, 0, i)
        eeg_container['layout'].addLayout(band_layout)
        graphs_layout.addLayout(eeg_container['layout'])

        # PPG Column
        ppg_container = self._create_graph_container(
            "❤️ PPG Signal",
            "PPG Waveform"
        )
        self.ppg_plot = ppg_container['plot']
        self.ppg_curve = self.ppg_plot.plot(pen=pg.mkPen('#E53935', width=2.5))
        self.ppg_peaks = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(255, 0, 0, 200))
        self.ppg_plot.addItem(self.ppg_peaks)

        # Metrics
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.setSpacing(10)
        self.hr_label = QtWidgets.QLabel("HR: --")
        self.hr_label.setStyleSheet("""
            font-size: 18px; 
            color: #00FF00; 
            font-weight: bold;
            background: #1a1a1a; 
            padding: 10px; 
            border-radius: 8px;
            border: 2px solid #00FF00;
        """)
        self.hrv_label = QtWidgets.QLabel("HRV: --")
        self.hrv_label.setStyleSheet("""
            font-size: 18px; 
            color: #FF4500; 
            font-weight: bold;
            background: #1a1a1a; 
            padding: 10px; 
            border-radius: 8px;
            border: 2px solid #FF4500;
        """)
        metrics_layout.addWidget(self.hr_label)
        metrics_layout.addWidget(self.hrv_label)
        ppg_container['layout'].addLayout(metrics_layout)
        graphs_layout.addLayout(ppg_container['layout'])

        # Stress Column
        stress_container = self._create_graph_container(
            "📊 Stress Index",
            "Stress Over Time"
        )
        self.stress_plot = stress_container['plot']
        self.stress_curve = self.stress_plot.plot(pen=pg.mkPen('#FFD700', width=3))

        self.stress_label = QtWidgets.QLabel("Stress: --")
        self.stress_label.setStyleSheet("""
            font-size: 18px; 
            color: #FFD700; 
            font-weight: bold;
            background: #1a1a1a; 
            padding: 10px; 
            border-radius: 8px;
            border: 2px solid #FFD700;
        """)
        stress_container['layout'].addWidget(self.stress_label)
        graphs_layout.addLayout(stress_container['layout'])

        main_layout.addLayout(graphs_layout)

        # ===== STATUS AND RECORDING INFO =====
        status_frame = QtWidgets.QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QtWidgets.QHBoxLayout(status_frame)

        self.status_label = QtWidgets.QLabel("⚪ Status: Ready to start")
        self.status_label.setStyleSheet("font-size: 14px; color: #FFF; font-weight: bold;")

        self.duration_label = QtWidgets.QLabel("⏱️ Duration: 0:00")
        self.duration_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")

        self.bp_status_label = QtWidgets.QLabel("🩺 BP: Not measured")
        self.bp_status_label.setStyleSheet("font-size: 14px; color: #FFF; font-weight: bold;")

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.duration_label)
        status_layout.addStretch()
        status_layout.addWidget(self.bp_status_label)

        main_layout.addWidget(status_frame)

        # ===== CONTROL BUTTONS =====
        button_layout = QtWidgets.QGridLayout()
        button_layout.setSpacing(10)

        self.btn_baseline_bp = self._create_button("🩺 1. Baseline BP", "#1976D2", True)
        self.btn_stream_start = self._create_button("▶ 2. Start Stream", "#388E3C", True)
        button_layout.addWidget(self.btn_baseline_bp, 0, 0)
        button_layout.addWidget(self.btn_stream_start, 0, 1)

        self.btn_start_recording = self._create_button("⏺ 3. Start Recording", "#D32F2F", True)
        self.btn_stop_recording = self._create_button("⏹ 4. Stop Recording", "#7B1FA2", True)
        button_layout.addWidget(self.btn_start_recording, 1, 0)
        button_layout.addWidget(self.btn_stop_recording, 1, 1)

        self.btn_stream_stop = self._create_button("⏹ 5. Stop Stream", "#C62828", True)
        self.btn_question_end_bp = self._create_button("💓 6. Question-End BP", "#F57C00", True)
        button_layout.addWidget(self.btn_stream_stop, 2, 0)
        button_layout.addWidget(self.btn_question_end_bp, 2, 1)

        self.btn_open_tlx = self._create_button("📋 Self Report", "#7B1FA2", True)


        button_layout.addWidget(self.btn_open_tlx, 3, 0)


        self.btn_exit = self._create_button("🚪 Exit Application", "#455A64", True)
        button_layout.addWidget(self.btn_exit, 4, 0, 1, 2)

        main_layout.addLayout(button_layout)

        # Connect signals
        self.btn_baseline_bp.clicked.connect(self.measure_baseline_bp)
        self.btn_stream_start.clicked.connect(self.start_stream)
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        self.btn_stream_stop.clicked.connect(self.stop_stream)
        self.btn_question_end_bp.clicked.connect(self.measure_question_end_bp)
        self.btn_open_tlx.clicked.connect(self.open_tlx_form)

        self.btn_exit.clicked.connect(self.exit_application)

    def _create_graph_container(self, title, plot_title):
        """Create a styled graph container"""
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setSpacing(8)

        label = QtWidgets.QLabel(title)
        label.setStyleSheet("""
            font-size: 15px; 
            font-weight: bold; 
            color: #FFF;
            background: #2a2a2a;
            padding: 8px;
            border-radius: 5px;
        """)
        container_layout.addWidget(label)

        plot = pg.PlotWidget(title=plot_title)
        plot.setBackground('#0d0d0d')
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel('bottom', 'Samples')
        container_layout.addWidget(plot)

        return {'layout': container_layout, 'plot': plot}

    def _create_button(self, text, color, enabled=True):
        """Create a styled button"""
        btn = QtWidgets.QPushButton(text)
        btn.setEnabled(enabled)
        btn.setMinimumHeight(50)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if enabled else QtCore.Qt.CursorShape.ForbiddenCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }}
            QPushButton:hover {{
                background: {color}DD;
            }}
            QPushButton:pressed {{
                background: {color}AA;
            }}
            QPushButton:disabled {{
                background: #37474F;
                color: #78909C;
            }}
        """)
        return btn

    def _update_duration(self):
        """Update recording duration display"""
        if self.recording and self.recording_start_time:
            elapsed = int(time.time() - self.recording_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.duration_label.setText(f"⏱️ Recording: {mins}:{secs:02d}")
            self.duration_label.setStyleSheet("font-size: 14px; color: #F44336; font-weight: bold;")
        elif not self.recording and self.recording_duration > 0:
            mins = self.recording_duration // 60
            secs = self.recording_duration % 60
            self.duration_label.setText(f"⏱️ Recorded: {mins}:{secs:02d}")
            self.duration_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
        else:
            self.duration_label.setText("⏱️ Duration: 0:00")
            self.duration_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")

    # ==================== BP MEASUREMENTS (UNCHANGED) ====================
    def measure_baseline_bp(self):
        """Measure baseline blood pressure"""
        self.status_label.setText("🩺 Measuring baseline BP...")
        self.bp_thread = BPWorker()
        self.bp_thread.reading.connect(self._save_baseline_bp)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def _save_baseline_bp(self, sys, dia, map_val, pulse):
        """Save baseline BP measurement"""
        self.base_bp = (sys, dia, pulse)
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, "Baseline", sys, dia, map_val, pulse, 0, 0, 0])
        self.bp_status_label.setText(f"✅ Baseline: {sys}/{dia} mmHg, Pulse: {pulse}")
        self.status_label.setText("✅ Baseline BP saved successfully")
        self.btn_open_tlx.setEnabled(True)

    def measure_question_end_bp(self):
        """Measure question-end blood pressure"""
        if not self.base_bp:
            QtWidgets.QMessageBox.warning(self, "⚠️ Error",
                                          "Please measure Baseline BP first!")
            return
        self.status_label.setText("💓 Measuring question-end BP...")
        self.bp_thread = BPWorker()
        self.bp_thread.reading.connect(self._save_question_end_bp)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def _save_question_end_bp(self, sys, dia, map_val, pulse):
        """Save question-end BP measurement"""
        delta_sys = sys - self.base_bp[0]
        delta_dia = dia - self.base_bp[1]
        delta_pulse = pulse - self.base_bp[2]
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, "Question-End", sys, dia, map_val, pulse,
                                    delta_sys, delta_dia, delta_pulse])
        self.question_end_bp = (sys, dia, pulse, delta_sys, delta_dia, delta_pulse)
        self.bp_status_label.setText(
            f"✅ Q-End: {sys}/{dia}, Pulse: {pulse} (Δ: {delta_sys:+d}/{delta_dia:+d}/{delta_pulse:+d})"
        )
        self.status_label.setText("✅ Question-End BP saved successfully")

    # ==================== NASA-TLX (UNCHANGED) ====================
    def open_tlx_form(self):

        dialog = NASATLXDialog(self)

        if dialog.exec() == QtWidgets.QDialog.Accepted:

            scores = dialog.get_scores()
            overall = dialog.get_overall_score()
            label = tlx_to_stress_label(overall)

            label_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

            tlx_file = os.path.join(self.folder, "self_report.csv")

            with open(tlx_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Dimension", "Score"])
                for k, v in scores.items():
                    writer.writerow([k, v])
                writer.writerow(["Final_Average", overall])
                writer.writerow(["Stress_Label", label])
                writer.writerow(["Stress_Level", label_names[label]])

            self.status_label.setText(
                f"Self-Report Score: {overall:.2f} → {label_names[label]}"
            )

            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Final Score: {overall:.2f}\nStress Level: {label_names[label]}"
            )

    # ==================== STREAM MANAGEMENT ====================
    def start_stream(self):
        """Start EEG and PPG streaming"""
        if self.streaming:
            return
        try:
            self.status_label.setText("🔌 Connecting to Muse device...")
            QtWidgets.QApplication.processEvents()

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "muselsl", "stream", "--ppg"],
                creationflags=creationflags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(6)

            eeg_streams = resolve_byprop("type", "EEG", timeout=10)
            ppg_streams = resolve_byprop("type", "PPG", timeout=10)

            if not eeg_streams or not ppg_streams:
                raise RuntimeError("❌ Streams not found. Check Muse connection.")

            self.eeg_inlet = StreamInlet(eeg_streams[0])
            self.ppg_inlet = StreamInlet(ppg_streams[0])
            self.ppg_channels = self.ppg_inlet.info().channel_count()
            self.ppg_buffer = np.zeros((self.ppg_channels, PPG_FS_EXPECTED * DISPLAY_SEC))

            self.streaming = True


            self.status_label.setText("✅ Stream connected - Using CHUNK-BASED sampling with LSL timestamps")
            self.btn_stream_start.setEnabled(False)
            self.btn_stream_stop.setEnabled(True)
            self.btn_start_recording.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"❌ Stream Error: {str(e)}")
            if self.proc:
                self.proc.terminate()
                self.proc = None
            QtWidgets.QMessageBox.critical(self, "❌ Error",
                                           f"Failed to start stream:\n{str(e)}")

    def stop_stream(self):
        """Stop EEG and PPG streaming"""
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

    def stop_recording(self):
        self.recording = False

        if hasattr(self, "eeg_f"):
            self.eeg_f.close()

        if hasattr(self, "ppg_f"):
            self.ppg_f.close()

        if self.recording_start_time:
            self.recording_duration = int(time.time() - self.recording_start_time)

        self.status_label.setText("⏹ Recording stopped and saved")
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)

        print("\n✅ Recording stopped safely")

    def start_recording(self):
        if not self.streaming:
            return

        self.recording = True
        self.recording_start_time = time.time()

        # Open CSV files ONCE (memory safe)
        self.eeg_f = open(self.eeg_file, "a", newline="")
        self.eeg_writer = csv.writer(self.eeg_f)

        self.ppg_f = open(self.ppg_file, "a", newline="")
        self.ppg_writer = csv.writer(self.ppg_f)

        self.first_eeg_timestamp = None
        self.first_ppg_timestamp = None
        self.hr_update_time = time.time()

        self.status_label.setText("⏺ Recording in progress...")
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(True)

        print("✅ Recording started (Direct Disk Write Mode)")



    # ==================== CORRECTED DATA UPDATE WITH CHUNK PULLING ====================
    def _update_data(self):

        if not self.streaming:
            return

        try:

            # ================= EEG =================
            if self.eeg_inlet:
                eeg_samples, eeg_timestamps = self.eeg_inlet.pull_chunk(timeout=0.0)

                if len(eeg_samples) > 0:

                    if self.recording and self.first_eeg_timestamp is None:
                        self.first_eeg_timestamp = eeg_timestamps[0]

                    num = len(eeg_samples)
                    samples_array = np.array(eeg_samples).T[:4]

                    self.eeg_buffer = np.roll(self.eeg_buffer, -num, axis=1)
                    self.eeg_buffer[:, -num:] = samples_array

                    if self.recording and self.first_eeg_timestamp is not None:
                        for sample, lsl_ts in zip(eeg_samples, eeg_timestamps):
                            relative_ts = lsl_ts - self.first_eeg_timestamp
                            self.eeg_writer.writerow([relative_ts] + sample[:4])

            # ================= PPG =================
            if self.ppg_inlet and self.ppg_buffer is not None:
                ppg_samples, ppg_timestamps = self.ppg_inlet.pull_chunk(timeout=0.0)

                if len(ppg_samples) > 0:

                    if self.recording and self.first_ppg_timestamp is None:
                        self.first_ppg_timestamp = ppg_timestamps[0]

                    num = len(ppg_samples)
                    samples_array = np.array(ppg_samples).T

                    self.ppg_buffer = np.roll(self.ppg_buffer, -num, axis=1)
                    self.ppg_buffer[:, -num:] = samples_array

                    # HR update once per second
                    if time.time() - self.hr_update_time > 1:
                        hr_temp, hrv_temp, _ = compute_ppg_hr_hrv(
                            self.ppg_buffer[0], PPG_FS_EXPECTED
                        )
                        if hr_temp > 0:
                            self.current_hr = hr_temp
                            self.current_hrv = hrv_temp
                        self.hr_update_time = time.time()

                    hr = self.current_hr
                    hrv = self.current_hrv

                    if self.recording and self.first_ppg_timestamp is not None:
                        for sample, lsl_ts in zip(ppg_samples, ppg_timestamps):
                            relative_ts = lsl_ts - self.first_ppg_timestamp

                            ppg_data = list(sample[:4])
                            while len(ppg_data) < 4:
                                ppg_data.append(0)

                            self.ppg_writer.writerow(
                                [relative_ts] + ppg_data + [hr, hrv]
                            )

        except Exception as e:
            print(f"Data update error: {e}")
            traceback.print_exc()

    def _update_graphs(self):
        """Update all graphs with current data"""
        if not self.streaming:
            return

        try:
            # EEG graphs
            for i, curve in enumerate(self.eeg_curves):
                signal = self.eeg_buffer[i]
                smoothed = smooth_signal(signal, window=EEG_SMOOTH_WINDOW)
                offset_signal = smoothed + (i * 30)
                curve.setData(offset_signal)

            # Band powers (using expected FS for display)
            if np.any(self.eeg_buffer):
                all_powers = {"Delta": 0, "Theta": 0, "Alpha": 0, "Beta": 0, "Gamma": 0}
                for ch in range(4):
                    powers = compute_band_powers(self.eeg_buffer[ch], EEG_FS_EXPECTED, include_gamma=True)
                    for band in all_powers:
                        all_powers[band] += powers[band]
                for band in all_powers:
                    all_powers[band] /= 4
                    self.band_labels[band].setText(f"{band}: {all_powers[band]:.2f}")

            # PPG graph
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

                hr, hrv, peaks = compute_ppg_hr_hrv(ppg_signal, PPG_FS_EXPECTED)

                if hr > 0:
                    self.hr_label.setText(f"HR: {hr:.1f} bpm")
                    self.hrv_label.setText(f"HRV: {hrv:.1f} ms")
                    self.hr_history.append(hr)
                    if len(self.hr_history) > 100:
                        self.hr_history.pop(0)

                # Stress calculation (for display only)
                if np.any(self.eeg_buffer):
                    powers = compute_band_powers(self.eeg_buffer[0], EEG_FS_EXPECTED, include_gamma=True)
                    beta_alpha = powers["Beta"] / (powers["Alpha"] + 1e-6)

                    if hrv > 0:
                        hrv_component = 1 / (hrv + 1e-6)
                    else:
                        hrv_component = 0

                    stress = 0.6 * beta_alpha + 0.4 * hrv_component
                    self.stress_label.setText(f"Stress: {stress:.2f}")
                    self.stress_history.append(stress)
                    if len(self.stress_history) > 100:
                        self.stress_history.pop(0)

                    stress_smooth = smooth_signal(np.array(self.stress_history),
                                                  window=STRESS_SMOOTH_WINDOW)
                    self.stress_curve.setData(stress_smooth)

                if len(peaks) > 0:
                    peak_y = ppg_smooth[peaks]
                    self.ppg_peaks.setData(peaks, peak_y)

        except Exception as e:
            print(f"Graph update error: {e}")

    def exit_application(self):
        """Exit the application"""
        if self.streaming:
            reply = QtWidgets.QMessageBox.question(
                self, "⚠️ Warning",
                "Stream is still active. Stop and exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.stop_stream()
            else:
                return

        QtWidgets.QApplication.quit()

    def closeEvent(self, event):
        """Handle window close event"""
        if self.streaming:
            self.stop_stream()
        event.accept()


# ==================== MAIN ====================
if __name__ == "__main__":


    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')



    # Dark theme
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15, 15, 15))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.GlobalColor.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(33, 150, 243))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(33, 150, 243))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.GlobalColor.black)
    app.setPalette(palette)

    window = BiosignalRecorder()
    window.show()

    sys.exit(app.exec())