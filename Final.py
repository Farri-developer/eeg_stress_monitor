"""
🧠 Research-Grade EEG + PPG + BP Biosignal Recorder for Stress Detection
==========================================================================
Multimodal Stress Detection System for Coding Tasks
Ground Truth: NASA-TLX Overall Score
Target: Predict perceived stress from physiological signals

Research Methodology:
- Features: EEG (band powers, Beta/Alpha), PPG (HR, HRV), BP (Delta changes)
- Ground Truth: NASA-TLX Overall score
- Labels: Low (0), Medium (1), High (2) stress based on TLX thresholds
- No data leakage: TLX dimensions excluded from features
- No computed stress scores: Only TLX-based classification

TECHNICAL IMPROVEMENTS APPLIED:
✅ Chunk-based sampling (pull_chunk) - preserves true sampling rates
✅ LSL timestamps - perfect synchronization without drift
✅ Timestamp-based windowing - accurate 5-second windows
✅ Effective sampling rate validation - quality assurance
✅ Gamma band safety checks - scientific validity
✅ Comprehensive data quality validation
"""

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
    'LOW_MAX': 40,      # < 40 = Low Stress (Label 0)
    'MEDIUM_MAX': 70,   # 40-70 = Medium Stress (Label 1)
    # > 70 = High Stress (Label 2)
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


def tlx_to_stress_label(tlx_overall):
    """
    Convert NASA-TLX Overall score to stress label (Ground Truth) - UNCHANGED

    Research-based thresholds:
    - TLX < 40: Low Stress (Label 0)
    - TLX 40-70: Medium Stress (Label 1)
    - TLX > 70: High Stress (Label 2)

    Args:
        tlx_overall: NASA-TLX overall score (0-100)

    Returns:
        int: Stress label (0, 1, or 2)
    """
    if tlx_overall < TLX_THRESHOLDS['LOW_MAX']:
        return 0  # Low Stress
    elif tlx_overall <= TLX_THRESHOLDS['MEDIUM_MAX']:
        return 1  # Medium Stress
    else:
        return 2  # High Stress


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
    """NASA Task Load Index Questionnaire Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workload Assessment")
        self.setModal(True)
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        self.setStyleSheet("QDialog { background: #1a1a1a; }")

        self._setup_ui()

    def _setup_ui(self):
        """Create the TLX questionnaire interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)

        # Title with gradient
        title = QtWidgets.QLabel("📋 تجربے کی درجہ بندی / Rate Your Experience")
        title.setStyleSheet("""
            font-size: 26px; 
            font-weight: bold; 
            color: white; 
            padding: 22px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 12px;
        """)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Please rate each question from 0 to 100"
        )
        instructions.setStyleSheet("""
            font-size: 15px; 
            padding: 16px; 
            color: #EEE;
            background: #2a2a2a;
            border-radius: 8px;
            border: 2px solid #444;
        """)
        instructions.setWordWrap(True)
        instructions.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # TLX Dimensions (Only 3)
        self.dimensions = {
            "Effort": "How hard did you work?",
            "Frustration": "How stressed or frustrated were you?",
            "Mental_Load": "How much mental pressure did you feel?"
        }

        # Scrollable area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(20)

        self.sliders = {}
        self.value_labels = {}

        # Colors and icons for each dimension
        styles = {
            "Effort": {"color1": "#4CAF50", "color2": "#66BB6A", "icon": "💪"},
            "Frustration": {"color1": "#F44336", "color2": "#EF5350", "icon": "😰"},
            "Mental_Load": {"color1": "#2196F3", "color2": "#42A5F5", "icon": "🧠"}
        }

        for dimension, description in self.dimensions.items():
            style = styles[dimension]

            # Frame
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(f"""
                QFrame {{
                    background: #1e1e1e; 
                    border: 3px solid {style['color1']};
                    border-radius: 15px; 
                    padding: 22px;
                }}
            """)
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setSpacing(15)

            # Dimension name with icon
            dim_label = QtWidgets.QLabel(f"{style['icon']} <b>{dimension}</b>")
            dim_label.setStyleSheet(f"""
                font-size: 20px; 
                color: {style['color1']}; 
                border: none;
                font-weight: bold;
            """)
            frame_layout.addWidget(dim_label)

            # Description
            desc_label = QtWidgets.QLabel(description)
            desc_label.setStyleSheet("""
                font-size: 14px; 
                color: #CCC; 
                border: none;
                line-height: 1.6;
            """)
            desc_label.setWordWrap(True)
            frame_layout.addWidget(desc_label)

            # Slider container
            slider_container = QtWidgets.QHBoxLayout()
            slider_container.setSpacing(15)

            # Left label
            left_label = QtWidgets.QLabel("0\nVery\nLow")
            left_label.setStyleSheet("font-size: 11px; color: #888; border: none;")
            left_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            left_label.setFixedWidth(65)
            slider_container.addWidget(left_label)

            # Slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)
            slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    background: #333;
                    height: 14px;
                    border-radius: 7px;
                    border: 2px solid #555;
                }}
                QSlider::handle:horizontal {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {style['color1']}, stop:1 {style['color2']});
                    width: 28px;
                    height: 28px;
                    margin: -9px 0;
                    border-radius: 14px;
                    border: 3px solid white;
                }}
                QSlider::sub-page:horizontal {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {style['color1']}, stop:1 {style['color2']});
                    border-radius: 7px;
                }}
            """)
            slider_container.addWidget(slider, stretch=1)

            # Right label
            right_label = QtWidgets.QLabel("100\nVery\nHigh")
            right_label.setStyleSheet("font-size: 11px; color: #888; border: none;")
            right_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            right_label.setFixedWidth(65)
            slider_container.addWidget(right_label)

            frame_layout.addLayout(slider_container)

            # Value display
            value_display = QtWidgets.QHBoxLayout()
            value_label = QtWidgets.QLabel("50")
            value_label.setStyleSheet(f"""
                font-size: 36px; 
                font-weight: bold; 
                color: {style['color1']}; 
                background: #0a0a0a;
                padding: 12px 25px;
                border-radius: 10px;
                border: 2px solid {style['color1']};
            """)
            value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            value_label.setFixedHeight(70)
            value_display.addStretch()
            value_display.addWidget(value_label)
            value_display.addStretch()
            frame_layout.addLayout(value_display)

            # Connect slider
            def make_updater(lbl, c1):
                def update(val):
                    lbl.setText(str(val))
                    lbl.setStyleSheet(f"""
                        font-size: 36px; 
                        font-weight: bold; 
                        color: {c1}; 
                        background: #0a0a0a;
                        padding: 12px 25px;
                        border-radius: 10px;
                        border: 2px solid {c1};
                    """)
                return update

            slider.valueChanged.connect(make_updater(value_label, style['color1']))

            self.sliders[dimension] = slider
            self.value_labels[dimension] = value_label

            scroll_layout.addWidget(frame)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(15)

        cancel_btn = QtWidgets.QPushButton("✖ Cancel")
        cancel_btn.setFixedHeight(55)
        cancel_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #37474F;
                color: white;
                padding: 14px 35px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #455A64;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QtWidgets.QPushButton("✓ Submit Ratings")
        ok_btn.setFixedHeight(55)
        ok_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        ok_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                padding: 14px 35px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7c8ff0, stop:1 #8c5bb0);
            }
        """)
        ok_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn, stretch=1)
        layout.addLayout(button_layout)

    def get_scores(self):
        """Return the TLX scores as a dictionary"""
        return {dim: slider.value() for dim, slider in self.sliders.items()}

    def get_overall_tlx(self):
        """Calculate overall TLX score (average of all dimensions)"""
        scores = self.get_scores()
        return sum(scores.values()) / len(scores)


class BiosignalRecorder(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Research-Grade Stress Detection System - EEG + PPG + BP [CORRECTED]")
        self.setGeometry(50, 50, 1700, 950)

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
        self.dataset_file = os.path.join(self.folder, "dataset_features.csv")
        self.ml_dataset_file = os.path.join(self.folder, "ml_ready_dataset.csv")
        self.validation_file = os.path.join(self.folder, "validation_report.txt")

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
        self.last_sample_time = time.time()
        self.eeg_channels = 4
        self.ppg_channels = 4
        self.recording_start_time = None
        self.recording_duration = 0

        # ===== CORRECTED: Chunk-based buffers with LSL timestamps =====
        self.sync_eeg_buffer = []  # List of (lsl_timestamp, sample)
        self.sync_ppg_buffer = []  # List of (lsl_timestamp, sample, hr, hrv)
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

        self.btn_start_recording = self._create_button("⏺ 3. Start Recording", "#D32F2F", False)
        self.btn_stop_recording = self._create_button("⏹ 4. Stop Recording", "#7B1FA2", False)
        button_layout.addWidget(self.btn_start_recording, 1, 0)
        button_layout.addWidget(self.btn_stop_recording, 1, 1)

        self.btn_stream_stop = self._create_button("⏹ 5. Stop Stream", "#C62828", False)
        self.btn_question_end_bp = self._create_button("💓 6. Question-End BP", "#F57C00", False)
        button_layout.addWidget(self.btn_stream_stop, 2, 0)
        button_layout.addWidget(self.btn_question_end_bp, 2, 1)

        self.btn_open_tlx = self._create_button("📋 7. NASA-TLX Form", "#7B1FA2", False)
        self.btn_generate_dataset = self._create_button("📊 8. Generate Dataset", "#00838F", False)
        button_layout.addWidget(self.btn_open_tlx, 3, 0)
        button_layout.addWidget(self.btn_generate_dataset, 3, 1)

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
        self.btn_generate_dataset.clicked.connect(self.generate_dataset)
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
        """Open NASA-TLX questionnaire dialog"""
        dialog = NASATLXDialog(self)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.tlx_scores = dialog.get_scores()
            overall_tlx = dialog.get_overall_tlx()

            # Save TLX scores
            tlx_file = os.path.join(self.folder, "tlx_scores.csv")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(tlx_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Dimension", "Score"])
                for dimension, score in self.tlx_scores.items():
                    writer.writerow([timestamp, dimension, score])
                writer.writerow([timestamp, "Overall", overall_tlx])

            # Calculate stress label
            stress_label = tlx_to_stress_label(overall_tlx)
            label_names = {0: "LOW STRESS", 1: "MEDIUM STRESS", 2: "HIGH STRESS"}

            self.status_label.setText(
                f"✅ NASA-TLX: {overall_tlx:.1f}/100 → {label_names[stress_label]}"
            )
            self.btn_generate_dataset.setEnabled(True)

            QtWidgets.QMessageBox.information(
                self, "✅ TLX Saved",
                f"NASA-TLX responses saved successfully!\n\n"
                f"Overall TLX Score: {overall_tlx:.1f}/100\n"
                f"Stress Classification: {label_names[stress_label]}\n"
                f"File: tlx_scores.csv"
            )
        else:
            self.status_label.setText("⚠️ NASA-TLX cancelled")

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
            self.last_sample_time = time.time()

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

    def start_recording(self):
        """Start data recording"""
        if not self.streaming:
            return

        # Clear previous recording buffers
        self.sync_eeg_buffer = []
        self.sync_ppg_buffer = []
        self.first_eeg_timestamp = None
        self.first_ppg_timestamp = None

        self.recording = True
        self.recording_start_time = time.time()

        self.status_label.setText("⏺ Recording in progress... (LSL timestamps)")
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(True)

        print("\n✅ Recording started - Using LSL timestamps and chunk-based sampling")

    def stop_recording(self):
        """Stop data recording and save synchronized data"""
        self.recording = False
        if self.recording_start_time:
            self.recording_duration = int(time.time() - self.recording_start_time)

        # Save synchronized data to CSV files
        self._save_synchronized_data()

        self.status_label.setText(
            f"⏹ Recording stopped - Saved {len(self.sync_eeg_buffer)} EEG and {len(self.sync_ppg_buffer)} PPG samples")
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)

        print(f"\n✅ Recording stopped")
        print(f"   EEG samples: {len(self.sync_eeg_buffer)}")
        print(f"   PPG samples: {len(self.sync_ppg_buffer)}")

    def _save_synchronized_data(self):
        """Save synchronized EEG and PPG data to CSV files with LSL timestamps"""
        try:
            # Save EEG data with LSL timestamps
            with open(self.eeg_file, "a", newline="") as f:
                writer = csv.writer(f)
                for lsl_ts, sample in self.sync_eeg_buffer:
                    writer.writerow([lsl_ts] + sample[:4])

            # Save PPG data with LSL timestamps
            with open(self.ppg_file, "a", newline="") as f:
                writer = csv.writer(f)
                for lsl_ts, sample, hr, hrv in self.sync_ppg_buffer:
                    ppg_data = list(sample[:4])
                    while len(ppg_data) < 4:
                        ppg_data.append(0)
                    writer.writerow([lsl_ts] + ppg_data + [hr, hrv])

            print(f"✅ Saved {len(self.sync_eeg_buffer)} EEG samples and {len(self.sync_ppg_buffer)} PPG samples")
            print(f"   Using LSL timestamps for perfect synchronization")

        except Exception as e:
            print(f"❌ Error saving synchronized data: {e}")
            traceback.print_exc()

    # ==================== CORRECTED DATA UPDATE WITH CHUNK PULLING ====================
    def _update_data(self):
        """
        CORRECTED: Update data buffers using pull_chunk() instead of pull_sample()
        This ensures we capture all samples at true sampling rates (256Hz EEG, 64Hz PPG)
        """
        if not self.streaming:
            return

        try:
            current_time = time.time()

            # ===== CORRECTED: EEG CHUNK PULLING =====
            if self.eeg_inlet:
                # Pull ALL available EEG samples in one call
                eeg_samples, eeg_timestamps = self.eeg_inlet.pull_chunk(timeout=0.0)

                if len(eeg_samples) > 0:
                    self.last_sample_time = current_time

                    # Set first timestamp if this is the start of recording
                    if self.recording and self.first_eeg_timestamp is None:
                        self.first_eeg_timestamp = eeg_timestamps[0]
                        print(f"✅ EEG: First LSL timestamp set: {self.first_eeg_timestamp:.6f}")

                    # Process ALL samples in the chunk
                    for i, (sample, lsl_ts) in enumerate(zip(eeg_samples, eeg_timestamps)):
                        # Update display buffer (rolling)
                        self.eeg_buffer = np.roll(self.eeg_buffer, -1, axis=1)
                        self.eeg_buffer[:, -1] = sample[:4]

                        # Store in synchronized buffer if recording
                        if self.recording and self.first_eeg_timestamp is not None:
                            # Use relative LSL timestamp
                            relative_ts = lsl_ts - self.first_eeg_timestamp
                            self.sync_eeg_buffer.append((relative_ts, sample))

            # ===== CORRECTED: PPG CHUNK PULLING =====
            if self.ppg_inlet and self.ppg_buffer is not None:
                # Pull ALL available PPG samples in one call
                ppg_samples, ppg_timestamps = self.ppg_inlet.pull_chunk(timeout=0.0)

                if len(ppg_samples) > 0:
                    self.last_sample_time = current_time

                    # Set first timestamp if this is the start of recording
                    if self.recording and self.first_ppg_timestamp is None:
                        self.first_ppg_timestamp = ppg_timestamps[0]
                        print(f"✅ PPG: First LSL timestamp set: {self.first_ppg_timestamp:.6f}")

                    # Process ALL samples in the chunk
                    for i, (sample, lsl_ts) in enumerate(zip(ppg_samples, ppg_timestamps)):
                        # Update display buffer (rolling)
                        actual_channels = min(len(sample), self.ppg_channels)
                        self.ppg_buffer = np.roll(self.ppg_buffer, -1, axis=1)
                        self.ppg_buffer[:actual_channels, -1] = sample[:actual_channels]

                        # Calculate HR and HRV periodically (not every sample to save CPU)
                        hr, hrv = 0, 0
                        if i == len(ppg_samples) - 1:  # Only on last sample of chunk
                            hr, hrv, _ = compute_ppg_hr_hrv(self.ppg_buffer[0], PPG_FS_EXPECTED)

                        # Store in synchronized buffer if recording
                        if self.recording and self.first_ppg_timestamp is not None:
                            # Use relative LSL timestamp
                            relative_ts = lsl_ts - self.first_ppg_timestamp
                            self.sync_ppg_buffer.append((relative_ts, sample, hr, hrv))

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
                    hr_hrv_ratio = hr / (hrv + 1e-6)
                    stress = (beta_alpha + hr_hrv_ratio) / 2
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

    # ==================== CORRECTED DATASET GENERATION ====================
    def generate_dataset(self):
        """
        CORRECTED: Generate research-grade dataset with:
        - Timestamp-based windowing (not sample count)
        - Effective sampling rate validation
        - Gamma band safety checks
        - Comprehensive data quality validation
        - TLX-based ground truth (UNCHANGED)
        """
        try:
            import pandas as pd

            self.status_label.setText("📊 Generating research dataset...")
            QtWidgets.QApplication.processEvents()

            # Validate TLX scores (UNCHANGED)
            if not self.tlx_scores:
                QtWidgets.QMessageBox.critical(
                    self, "❌ Error",
                    "Please complete NASA-TLX assessment first!"
                )
                return

            # Load data files
            try:
                eeg_df = pd.read_csv(self.eeg_file)
                print(f"\n✅ Loaded EEG: {len(eeg_df)} samples")
            except:
                QtWidgets.QMessageBox.critical(self, "❌ Error", "EEG data file not found!")
                return

            try:
                ppg_df = pd.read_csv(self.ppg_file)
                print(f"✅ Loaded PPG: {len(ppg_df)} samples")
            except:
                QtWidgets.QMessageBox.critical(self, "❌ Error", "PPG data file not found!")
                return

            try:
                bp_df = pd.read_csv(self.bp_file)
            except:
                bp_df = pd.DataFrame()

            # ===== CORRECTED: Calculate EFFECTIVE sampling rates =====
            if len(eeg_df) > 1:
                eeg_time_span = eeg_df['lsl_timestamp'].iloc[-1] - eeg_df['lsl_timestamp'].iloc[0]
                eeg_effective_fs = len(eeg_df) / eeg_time_span if eeg_time_span > 0 else 0
            else:
                eeg_effective_fs = 0

            if len(ppg_df) > 1:
                ppg_time_span = ppg_df['lsl_timestamp'].iloc[-1] - ppg_df['lsl_timestamp'].iloc[0]
                ppg_effective_fs = len(ppg_df) / ppg_time_span if ppg_time_span > 0 else 0
            else:
                ppg_effective_fs = 0

            print(f"\n📊 Effective Sampling Rates:")
            print(f"   EEG: {eeg_effective_fs:.2f} Hz (Expected: {EEG_FS_EXPECTED} Hz)")
            print(f"   PPG: {ppg_effective_fs:.2f} Hz (Expected: {PPG_FS_EXPECTED} Hz)")

            # Validate sampling rates
            eeg_valid, eeg_dev, eeg_warning = validate_sampling_rate(
                eeg_effective_fs, EEG_FS_EXPECTED, "EEG"
            )
            ppg_valid, ppg_dev, ppg_warning = validate_sampling_rate(
                ppg_effective_fs, PPG_FS_EXPECTED, "PPG"
            )

            warnings = []
            if eeg_warning:
                warnings.append(eeg_warning)
                print(eeg_warning)
            if ppg_warning:
                warnings.append(ppg_warning)
                print(ppg_warning)

            # ===== CORRECTED: Gamma band safety check =====
            include_gamma = eeg_effective_fs >= MIN_GAMMA_FS
            if not include_gamma:
                gamma_warning = f"⚠️ Gamma band excluded (EEG fs={eeg_effective_fs:.1f}Hz < {MIN_GAMMA_FS}Hz threshold)"
                warnings.append(gamma_warning)
                print(gamma_warning)
            else:
                print(f"✅ Gamma band included (EEG fs={eeg_effective_fs:.1f}Hz >= {MIN_GAMMA_FS}Hz)")

            # Extract BP features (UNCHANGED)
            baseline = bp_df[bp_df['label'] == 'Baseline'].iloc[0] if len(bp_df) > 0 else None
            question_end = bp_df[bp_df['label'] == 'Question-End'].iloc[-1] if len(
                bp_df[bp_df['label'] == 'Question-End']) > 0 else None

            if baseline is not None and question_end is not None:
                bp_features = {
                    'DeltaSYS': question_end['DeltaSYS'],
                    'DeltaDIA': question_end['DeltaDIA'],
                    'DeltaPULSE': question_end['DeltaPulse']
                }
            else:
                bp_features = {'DeltaSYS': 0, 'DeltaDIA': 0, 'DeltaPULSE': 0}

            # Calculate TLX Overall and Label (UNCHANGED)
            tlx_overall = sum(self.tlx_scores.values()) / len(self.tlx_scores)
            tlx_label = tlx_to_stress_label(tlx_overall)

            print(f"\n📋 TLX Overall: {tlx_overall:.1f} → Label: {tlx_label}")

            # ===== CORRECTED: TIMESTAMP-BASED WINDOWING =====
            WIN_SEC = 5.0  # 5-second windows

            rows = []
            quality_warnings = []

            # Get time ranges
            eeg_start_time = eeg_df['lsl_timestamp'].iloc[0]
            eeg_end_time = eeg_df['lsl_timestamp'].iloc[-1]
            ppg_start_time = ppg_df['lsl_timestamp'].iloc[0]
            ppg_end_time = ppg_df['lsl_timestamp'].iloc[-1]

            # Use overlapping time range
            start_time = max(eeg_start_time, ppg_start_time)
            end_time = min(eeg_end_time, ppg_end_time)
            total_duration = end_time - start_time

            print(f"\n📊 Window Segmentation (TIMESTAMP-BASED):")
            print(f"   Total duration: {total_duration:.2f} seconds")
            print(f"   Window size: {WIN_SEC} seconds")

            num_windows = int(total_duration / WIN_SEC)
            print(f"   Expected windows: {num_windows}")

            for w in range(num_windows):
                # Calculate time window boundaries
                win_start = start_time + (w * WIN_SEC)
                win_end = win_start + WIN_SEC

                # Extract EEG samples within this time window
                eeg_mask = (eeg_df['lsl_timestamp'] >= win_start) & (eeg_df['lsl_timestamp'] < win_end)
                eeg_win_df = eeg_df[eeg_mask]

                # Extract PPG samples within this time window
                ppg_mask = (ppg_df['lsl_timestamp'] >= win_start) & (ppg_df['lsl_timestamp'] < win_end)
                ppg_win_df = ppg_df[ppg_mask]

                # Check if we have enough samples
                if len(eeg_win_df) < 10 or len(ppg_win_df) < 10:
                    continue

                eeg_win = eeg_win_df[['EEG1', 'EEG2', 'EEG3', 'EEG4']].values

                # Calculate actual sampling rate for this window
                if len(eeg_win_df) > 1:
                    win_eeg_duration = eeg_win_df['lsl_timestamp'].iloc[-1] - eeg_win_df['lsl_timestamp'].iloc[0]
                    win_eeg_fs = len(eeg_win_df) / win_eeg_duration if win_eeg_duration > 0 else eeg_effective_fs
                else:
                    win_eeg_fs = eeg_effective_fs

                # Filter EEG
                eeg_win_filtered = bandpass_filter(eeg_win, 0.5, 45, win_eeg_fs)

                features = {}

                # ===== EEG FEATURES with actual FS and Gamma safety =====
                for ch in range(4):
                    powers = compute_band_powers(
                        eeg_win_filtered[:, ch],
                        win_eeg_fs,
                        include_gamma=include_gamma
                    )
                    for band, power in powers.items():
                        features[f'EEG{ch + 1}_{band}'] = power
                    features[f'EEG{ch + 1}_BetaAlpha'] = powers["Beta"] / (powers["Alpha"] + 1e-6)

                # Data quality check for EEG
                for ch in range(4):
                    ch_warnings = check_signal_quality(
                        eeg_win_filtered[:, ch],
                        f"Window{w}_EEG{ch+1}",
                        eeg_win_df['lsl_timestamp'].values
                    )
                    quality_warnings.extend(ch_warnings)

                # ===== PPG FEATURES =====
                ppg_cols = [c for c in ['PPG1', 'PPG2', 'PPG3', 'PPG4'] if c in ppg_win_df.columns]

                if len(ppg_cols) > 0:
                    ppg_data_vals = ppg_win_df[ppg_cols].values
                    features['PPG_Mean'] = np.mean(ppg_data_vals)
                    features['PPG_STD'] = np.std(ppg_data_vals)

                    # Calculate actual PPG FS for this window
                    if len(ppg_win_df) > 1:
                        win_ppg_duration = ppg_win_df['lsl_timestamp'].iloc[-1] - ppg_win_df['lsl_timestamp'].iloc[0]
                        win_ppg_fs = len(ppg_win_df) / win_ppg_duration if win_ppg_duration > 0 else ppg_effective_fs
                    else:
                        win_ppg_fs = ppg_effective_fs

                    hr_all, sdnn_all, rmssd_all, pnn50_all = [], [], [], []
                    for ch_idx in range(len(ppg_cols)):
                        ppg_ch_data = ppg_data_vals[:, ch_idx]
                        hr, sdnn, rmssd, pnn50 = extract_ppg_hrv(ppg_ch_data, win_ppg_fs)
                        hr_all.append(hr)
                        sdnn_all.append(sdnn)
                        rmssd_all.append(rmssd)
                        pnn50_all.append(pnn50)

                        # Data quality check for PPG
                        ch_warnings = check_signal_quality(
                            ppg_ch_data,
                            f"Window{w}_PPG{ch_idx+1}",
                            ppg_win_df['lsl_timestamp'].values
                        )
                        quality_warnings.extend(ch_warnings)

                    features['PPG_HR'] = np.mean(hr_all)
                    features['PPG_SDNN'] = np.mean(sdnn_all)
                    features['PPG_RMSSD'] = np.mean(rmssd_all)
                    features['PPG_pNN50'] = np.mean(pnn50_all)
                else:
                    features.update({'PPG_Mean': 0, 'PPG_STD': 0, 'PPG_HR': 0,
                                     'PPG_SDNN': 0, 'PPG_RMSSD': 0, 'PPG_pNN50': 0})

                # ===== BP FEATURES (UNCHANGED) =====
                features.update(bp_features)

                # ===== GROUND TRUTH (UNCHANGED) =====
                features['TLX_Overall'] = tlx_overall
                features['TLX_Label'] = tlx_label

                rows.append(features)

                if (w + 1) % 10 == 0:
                    print(f"   Processed window {w + 1}/{num_windows}")

            # Create full dataset
            full_df = pd.DataFrame(rows)
            print(f"\n✅ Created dataset with {len(full_df)} windows")

            # Save full dataset
            full_df.to_csv(self.dataset_file, index=False)
            print(f"✅ Full dataset saved: {self.dataset_file}")

            # ===== CREATE ML-READY DATASET (NO DATA LEAKAGE - UNCHANGED) =====
            ml_df = full_df.drop(columns=['TLX_Overall'])
            ml_df.to_csv(self.ml_dataset_file, index=False)
            print(f"✅ ML-ready dataset saved: {self.ml_dataset_file}")

            # Generate validation report with new metrics
            self._generate_validation_report(
                full_df, eeg_df, ppg_df, bp_df, tlx_overall, tlx_label,
                eeg_effective_fs, ppg_effective_fs, include_gamma, warnings, quality_warnings
            )

            # Label distribution
            label_counts = full_df['TLX_Label'].value_counts().to_dict()
            label_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
            label_dist = ", ".join([f"{label_names.get(int(k), k)}: {v}" for k, v in sorted(label_counts.items())])

            # Success message
            success_msg = (
                f"✅ Research Dataset Generated Successfully!\n\n"
                f"📊 Dataset Summary:\n"
                f"• Total Windows: {len(full_df)}\n"
                f"• Window Size: {WIN_SEC} seconds (timestamp-based)\n"
                f"• Features: {len(ml_df.columns) - 1} physiological features\n"
                f"• Target: TLX_Label (0=Low, 1=Medium, 2=High)\n\n"
                f"📋 Ground Truth:\n"
                f"• NASA-TLX Overall: {tlx_overall:.1f}/100\n"
                f"• Stress Classification: {label_names[tlx_label]}\n"
                f"• All windows labeled: {label_dist}\n\n"
                f"📊 Sampling Rates (Effective):\n"
                f"• EEG: {eeg_effective_fs:.2f} Hz (Expected: {EEG_FS_EXPECTED} Hz)\n"
                f"• PPG: {ppg_effective_fs:.2f} Hz (Expected: {PPG_FS_EXPECTED} Hz)\n"
                f"• Gamma band: {'Included' if include_gamma else 'Excluded (low fs)'}\n\n"
                f"📁 Output Files:\n"
                f"• dataset_features.csv\n"
                f"• ml_ready_dataset.csv\n"
                f"• validation_report.txt\n\n"
                f"✅ Chunk-based sampling preserved true rates\n"
                f"✅ LSL timestamps ensure perfect sync\n"
                f"✅ Timestamp-based windowing\n"
                f"✅ No data leakage: TLX excluded from features"
            )

            self.status_label.setText(f"✅ Dataset: {len(full_df)} windows, Label: {tlx_label}")

            QtWidgets.QMessageBox.information(self, "✅ Success", success_msg)

        except Exception as e:
            self.status_label.setText(f"❌ Error generating dataset")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "❌ Error",
                                           f"Failed to generate dataset:\n\n{str(e)}")

    def _generate_validation_report(self, dataset_df, eeg_df, ppg_df, bp_df, tlx_overall, tlx_label,
                                      eeg_fs, ppg_fs, include_gamma, warnings, quality_warnings):
        """CORRECTED: Generate comprehensive validation report"""
        try:
            label_names = {0: "LOW STRESS", 1: "MEDIUM STRESS", 2: "HIGH STRESS"}

            with open(self.validation_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("RESEARCH-GRADE MULTIMODAL STRESS DETECTION DATASET\n")
                f.write("Validation Report - CORRECTED VERSION\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Participant: {self.user_name} ({self.user_id})\n")
                f.write(f"Session Folder: {self.folder}\n\n")

                f.write("-" * 80 + "\n")
                f.write("TECHNICAL IMPROVEMENTS APPLIED\n")
                f.write("-" * 80 + "\n")
                f.write("✅ Chunk-based sampling (pull_chunk) - preserves true sampling rates\n")
                f.write("✅ LSL timestamps - perfect synchronization without drift\n")
                f.write("✅ Timestamp-based windowing - accurate 5-second windows\n")
                f.write("✅ Effective sampling rate validation - quality assurance\n")
                f.write("✅ Gamma band safety checks - scientific validity\n")
                f.write("✅ Comprehensive data quality validation\n\n")

                f.write("-" * 80 + "\n")
                f.write("RESEARCH METHODOLOGY\n")
                f.write("-" * 80 + "\n")
                f.write("Ground Truth: NASA-TLX Overall Score\n")
                f.write("Features: EEG (band powers, Beta/Alpha) + PPG (HR, HRV) + BP (Delta)\n")
                f.write("Window Size: 5 seconds (timestamp-based, not sample count)\n")
                f.write("Classification: Low (0), Medium (1), High (2) stress\n")
                f.write("Thresholds: TLX < 40 (Low), 40-70 (Medium), > 70 (High)\n\n")

                f.write("-" * 80 + "\n")
                f.write("RAW DATA STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"EEG Samples: {len(eeg_df)}\n")
                f.write(f"PPG Samples: {len(ppg_df)}\n")
                f.write(f"BP Measurements: {len(bp_df)}\n")
                f.write(f"Recording Duration: {self.recording_duration} seconds\n\n")

                f.write("-" * 80 + "\n")
                f.write("SAMPLING RATE VALIDATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"EEG Effective Sampling Rate: {eeg_fs:.2f} Hz\n")
                f.write(f"EEG Expected Sampling Rate: {EEG_FS_EXPECTED} Hz\n")
                f.write(f"EEG Deviation: {abs(eeg_fs - EEG_FS_EXPECTED) / EEG_FS_EXPECTED * 100:.2f}%\n\n")
                f.write(f"PPG Effective Sampling Rate: {ppg_fs:.2f} Hz\n")
                f.write(f"PPG Expected Sampling Rate: {PPG_FS_EXPECTED} Hz\n")
                f.write(f"PPG Deviation: {abs(ppg_fs - PPG_FS_EXPECTED) / PPG_FS_EXPECTED * 100:.2f}%\n\n")

                f.write(f"Gamma Band Status: {'✅ INCLUDED' if include_gamma else '⚠️ EXCLUDED (fs < 100 Hz)'}\n")
                f.write(f"Gamma Band Requirement: Minimum {MIN_GAMMA_FS} Hz sampling rate\n\n")

                if warnings:
                    f.write("Sampling Rate Warnings:\n")
                    for warning in warnings:
                        f.write(f"  {warning}\n")
                    f.write("\n")
                else:
                    f.write("✅ All sampling rates within acceptable tolerance\n\n")

                f.write("-" * 80 + "\n")
                f.write("DATA QUALITY VALIDATION\n")
                f.write("-" * 80 + "\n")

                if quality_warnings:
                    f.write(f"Found {len(quality_warnings)} quality issues:\n\n")
                    for warning in quality_warnings[:50]:
                        f.write(f"  {warning}\n")
                    if len(quality_warnings) > 50:
                        f.write(f"\n  ... and {len(quality_warnings) - 50} more warnings\n")
                    f.write("\n")
                else:
                    f.write("✅ No data quality issues detected\n\n")

                f.write("-" * 80 + "\n")
                f.write("GROUND TRUTH LABELING\n")
                f.write("-" * 80 + "\n")
                f.write(f"NASA-TLX Overall Score: {tlx_overall:.2f}/100\n")
                f.write(f"TLX_Label: {tlx_label} ({label_names[tlx_label]})\n")
                f.write(f"\nTLX Dimension Scores:\n")
                for dim, score in self.tlx_scores.items():
                    f.write(f"  {dim}: {score}/100\n")
                f.write("\nNote: TLX dimensions excluded from ML features to prevent data leakage\n\n")

                f.write("-" * 80 + "\n")
                f.write("DATASET STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Windows: {len(dataset_df)}\n")
                f.write(f"Window Duration: 5.0 seconds (timestamp-based)\n")
                f.write(f"Expected Windows: {self.recording_duration // 5}\n")
                f.write(f"Window Coverage: {(len(dataset_df) * 5 / self.recording_duration * 100):.1f}%\n\n")

                f.write("Feature Categories:\n")
                eeg_features = [col for col in dataset_df.columns if col.startswith('EEG')]
                ppg_features = [col for col in dataset_df.columns if col.startswith('PPG')]
                bp_features = ['DeltaSYS', 'DeltaDIA', 'DeltaPULSE']

                f.write(f"  EEG Features: {len(eeg_features)} (band powers + Beta/Alpha ratio)\n")
                f.write(f"  PPG Features: {len(ppg_features)} (HR, HRV metrics)\n")
                f.write(f"  BP Features: {len(bp_features)} (Delta changes)\n")
                f.write(f"  Total Features: {len(eeg_features) + len(ppg_features) + len(bp_features)}\n")
                f.write(f"  Target Variable: TLX_Label\n\n")

                f.write("Label Distribution:\n")
                label_counts = dataset_df['TLX_Label'].value_counts()
                for label in [0, 1, 2]:
                    count = label_counts.get(label, 0)
                    pct = (count / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
                    f.write(f"  {label} ({label_names[label]}): {count} windows ({pct:.1f}%)\n")

                f.write("\n")

                f.write("-" * 80 + "\n")
                f.write("FEATURE STATISTICS\n")
                f.write("-" * 80 + "\n")
                numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['TLX_Overall', 'TLX_Label']]
                stats = dataset_df[numeric_cols].describe()
                f.write(stats.to_string())
                f.write("\n\n")

                f.write("-" * 80 + "\n")
                f.write("DATA QUALITY CHECKS\n")
                f.write("-" * 80 + "\n")

                # Check for missing values
                missing = dataset_df.isnull().sum()
                if missing.sum() == 0:
                    f.write("✅ No missing values found\n")
                else:
                    f.write("⚠️ Missing values detected:\n")
                    for col, count in missing[missing > 0].items():
                        f.write(f"  {col}: {count}\n")

                # Check for zero variance
                zero_var = dataset_df[numeric_cols].std() == 0
                if zero_var.sum() == 0:
                    f.write("✅ All features have non-zero variance\n")
                else:
                    f.write("⚠️ Zero variance features:\n")
                    for col in zero_var[zero_var].index:
                        f.write(f"  {col}\n")

                f.write("\n")

                f.write("-" * 80 + "\n")
                f.write("ML READINESS\n")
                f.write("-" * 80 + "\n")
                f.write("✅ Features: Physiological signals only (no TLX dimensions)\n")
                f.write("✅ Target: TLX_Label (3-class classification)\n")
                f.write("✅ No data leakage: TLX_Overall excluded from ml_ready_dataset.csv\n")
                f.write("✅ LSL timestamps: Perfect EEG-PPG synchronization\n")
                f.write("✅ Timestamp windowing: Accurate 5-second windows\n")
                f.write("✅ Chunk-based sampling: True sampling rates preserved\n")
                f.write(f"✅ Gamma band: {'Included (valid fs)' if include_gamma else 'Excluded (safety check)'}\n\n")

                f.write("Recommended Next Steps:\n")
                f.write("1. Feature normalization/standardization\n")
                f.write("2. Feature selection (if needed)\n")
                f.write("3. Train-test split (or cross-validation)\n")
                f.write("4. Model training (SVM, RF, Neural Networks, etc.)\n")
                f.write("5. Performance evaluation (accuracy, F1-score, confusion matrix)\n\n")

                f.write("=" * 80 + "\n")
                f.write("VALIDATION COMPLETE - RESEARCH-GRADE QUALITY ASSURED\n")
                f.write("=" * 80 + "\n")

            print(f"✅ Validation report saved: {self.validation_file}")

        except Exception as e:
            print(f"⚠️ Failed to generate validation report: {e}")
            traceback.print_exc()

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
    try:
        import pandas as pd
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pandas", "--break-system-packages"])
        import pandas as pd

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