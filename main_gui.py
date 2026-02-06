"""
🧠 Complete EEG + PPG + BP Biosignal Recorder with NASA-TLX (FIXED)
====================================================================
Fixed Issues:
- EEG and PPG now record with synchronized timestamps
- Dataset generation creates correct number of windows
- Proper time alignment between all signals
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
EEG_FS = 256  # Muse EEG sampling rate
PPG_FS = 64   # Muse PPG sampling rate
DISPLAY_SEC = 10
STREAM_TIMEOUT = 30

# Smoothing parameters
EEG_SMOOTH_WINDOW = 15
PPG_SMOOTH_WINDOW = 5
STRESS_SMOOTH_WINDOW = 10


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
        self.setWindowTitle("🧠 Biosignal Recorder - EEG + PPG + BP + NASA-TLX (FIXED)")
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

        # NEW: Synchronized data buffers for recording
        self.sync_eeg_buffer = []
        self.sync_ppg_buffer = []
        self.recording_base_time = None

        # Data buffers for display
        self.eeg_buffer = np.zeros((4, EEG_FS * DISPLAY_SEC))
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

    # ==================== BP MEASUREMENTS ====================
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

    # ==================== NASA-TLX ====================
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

            self.status_label.setText(f"✅ NASA-TLX completed (Overall: {overall_tlx:.1f}/100)")
            self.btn_generate_dataset.setEnabled(True)

            QtWidgets.QMessageBox.information(
                self, "✅ TLX Saved",
                f"NASA-TLX responses saved successfully!\n\n"
                f"Overall TLX Score: {overall_tlx:.1f}/100\n"
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
            self.ppg_buffer = np.zeros((self.ppg_channels, PPG_FS * DISPLAY_SEC))

            self.streaming = True
            self.last_sample_time = time.time()

            self.status_label.setText("✅ Stream connected - Graphs updating in real-time")
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

        self.recording = True
        self.recording_start_time = time.time()
        self.recording_base_time = time.time()  # Base time for synchronization

        self.status_label.setText("⏺ Recording in progress...")
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(True)

    def stop_recording(self):
        """Stop data recording and save synchronized data"""
        self.recording = False
        if self.recording_start_time:
            self.recording_duration = int(time.time() - self.recording_start_time)

        # Save synchronized data to CSV files
        self._save_synchronized_data()

        self.status_label.setText(f"⏹ Recording stopped - Saved {len(self.sync_eeg_buffer)} EEG and {len(self.sync_ppg_buffer)} PPG samples")
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)

    def _save_synchronized_data(self):
        """Save synchronized EEG and PPG data to CSV files"""
        try:
            # Save EEG data
            with open(self.eeg_file, "a", newline="") as f:
                writer = csv.writer(f)
                for timestamp, sample in self.sync_eeg_buffer:
                    writer.writerow([timestamp] + sample[:4])

            # Save PPG data
            with open(self.ppg_file, "a", newline="") as f:
                writer = csv.writer(f)
                for timestamp, sample, hr, hrv in self.sync_ppg_buffer:
                    ppg_data = list(sample[:4])
                    while len(ppg_data) < 4:
                        ppg_data.append(0)
                    writer.writerow([timestamp] + ppg_data + [hr, hrv])

            print(f"✅ Saved {len(self.sync_eeg_buffer)} EEG samples and {len(self.sync_ppg_buffer)} PPG samples")

        except Exception as e:
            print(f"❌ Error saving synchronized data: {e}")

    # ==================== DATA UPDATE ====================
    def _update_data(self):
        """Update data buffers from streams with synchronization"""
        if not self.streaming:
            return

        try:
            current_time = time.time()

            # EEG data
            if self.eeg_inlet:
                eeg_sample, eeg_lsl_ts = self.eeg_inlet.pull_sample(timeout=0.0)
                if eeg_sample:
                    self.last_sample_time = current_time

                    # Update display buffer
                    self.eeg_buffer = np.roll(self.eeg_buffer, -1, axis=1)
                    self.eeg_buffer[:, -1] = eeg_sample[:4]

                    # Store in synchronized buffer if recording
                    if self.recording and self.recording_base_time:
                        relative_time = current_time - self.recording_base_time
                        self.sync_eeg_buffer.append((relative_time, eeg_sample))

            # PPG data
            if self.ppg_inlet and self.ppg_buffer is not None:
                ppg_sample, ppg_lsl_ts = self.ppg_inlet.pull_sample(timeout=0.0)
                if ppg_sample:
                    self.last_sample_time = current_time

                    # Update display buffer
                    actual_channels = min(len(ppg_sample), self.ppg_channels)
                    self.ppg_buffer = np.roll(self.ppg_buffer, -1, axis=1)
                    self.ppg_buffer[:actual_channels, -1] = ppg_sample[:actual_channels]

                    # Calculate HR and HRV
                    hr, hrv, peaks = compute_ppg_hr_hrv(self.ppg_buffer[0], PPG_FS)

                    # Store in synchronized buffer if recording
                    if self.recording and self.recording_base_time:
                        relative_time = current_time - self.recording_base_time
                        self.sync_ppg_buffer.append((relative_time, ppg_sample, hr, hrv))

        except Exception as e:
            print(f"Data update error: {e}")

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

            # Band powers
            if np.any(self.eeg_buffer):
                all_powers = {"Delta": 0, "Theta": 0, "Alpha": 0, "Beta": 0, "Gamma": 0}
                for ch in range(4):
                    powers = compute_band_powers(self.eeg_buffer[ch], EEG_FS)
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

                hr, hrv, peaks = compute_ppg_hr_hrv(ppg_signal, PPG_FS)

                if hr > 0:
                    self.hr_label.setText(f"HR: {hr:.1f} bpm")
                    self.hrv_label.setText(f"HRV: {hrv:.1f} ms")
                    self.hr_history.append(hr)
                    if len(self.hr_history) > 100:
                        self.hr_history.pop(0)

                # Stress calculation
                if np.any(self.eeg_buffer):
                    powers = compute_band_powers(self.eeg_buffer[0], EEG_FS)
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

    # ==================== DATASET GENERATION ====================
    def generate_dataset(self):
        """Generate dataset with proper window calculation"""
        try:
            import pandas as pd

            self.status_label.setText("📊 Generating dataset...")
            QtWidgets.QApplication.processEvents()

            # Load data files
            try:
                eeg_df = pd.read_csv(self.eeg_file)
                print(f"✅ Loaded EEG: {len(eeg_df)} samples")
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

            # Validate minimum data
            if len(eeg_df) < 100:
                QtWidgets.QMessageBox.warning(self, "⚠️ Warning",
                                              f"Only {len(eeg_df)} EEG samples (need 100+)")

            # Extract BP features
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

            # Process windows - FIXED CALCULATION
            WIN_SEC = 5
            WIN_SAMPLES_EEG = EEG_FS * WIN_SEC  # 256 * 5 = 1280 samples per window
            WIN_SAMPLES_PPG = PPG_FS * WIN_SEC  # 64 * 5 = 320 samples per window

            rows = []

            # Calculate number of complete windows
            num_windows_eeg = len(eeg_df) // WIN_SAMPLES_EEG
            num_windows_ppg = len(ppg_df) // WIN_SAMPLES_PPG
            num_windows = min(num_windows_eeg, num_windows_ppg)

            print(f"📊 Processing {num_windows} windows (EEG samples: {len(eeg_df)}, PPG samples: {len(ppg_df)})")
            print(f"   Window size: {WIN_SEC}s = {WIN_SAMPLES_EEG} EEG samples, {WIN_SAMPLES_PPG} PPG samples")

            for w in range(num_windows):
                # EEG window indices
                eeg_start = w * WIN_SAMPLES_EEG
                eeg_end = eeg_start + WIN_SAMPLES_EEG

                # PPG window indices
                ppg_start = w * WIN_SAMPLES_PPG
                ppg_end = ppg_start + WIN_SAMPLES_PPG

                # Extract windows
                eeg_win = eeg_df.iloc[eeg_start:eeg_end][['EEG1', 'EEG2', 'EEG3', 'EEG4']].values
                ppg_win = ppg_df.iloc[ppg_start:ppg_end]

                # Filter EEG
                eeg_win_filtered = bandpass_filter(eeg_win, 0.5, 45, EEG_FS)

                features = {'Window': w}

                # EEG features
                for ch in range(4):
                    powers = compute_band_powers(eeg_win_filtered[:, ch], EEG_FS)
                    for band, power in powers.items():
                        features[f'EEG{ch + 1}_{band}'] = power
                    features[f'EEG{ch + 1}_BetaAlpha'] = powers["Beta"] / (powers["Alpha"] + 1e-6)

                # PPG features
                ppg_cols = [c for c in ['PPG1', 'PPG2', 'PPG3', 'PPG4'] if c in ppg_win.columns]

                if len(ppg_cols) > 0:
                    ppg_data_vals = ppg_win[ppg_cols].values
                    features['PPG_Mean'] = np.mean(ppg_data_vals)
                    features['PPG_STD'] = np.std(ppg_data_vals)

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
                    features.update({'PPG_Mean': 0, 'PPG_STD': 0, 'PPG_HR': 0,
                                     'PPG_SDNN': 0, 'PPG_RMSSD': 0, 'PPG_pNN50': 0})

                features.update(bp_features)
                rows.append(features)

                if (w + 1) % 10 == 0:
                    print(f"   Processed window {w + 1}/{num_windows}")

            final_df = pd.DataFrame(rows)
            print(f"✅ Created dataset with {len(final_df)} windows")

            # Calculate stress score
            def calculate_stress_score(row):
                beta_alpha_cols = [f'EEG{i}_BetaAlpha' for i in range(1, 5)]
                mean_betaalpha = np.mean([row[col] for col in beta_alpha_cols])
                score = row['PPG_HR'] + (1 / (row['PPG_RMSSD'] + 1e-6)) + mean_betaalpha
                return score

            final_df['StressScore'] = final_df.apply(calculate_stress_score, axis=1)

            # Assign stress labels
            final_df['Stress_Label'] = pd.qcut(
                final_df['StressScore'],
                q=3,
                labels=['0', '1', '2'],
                duplicates='drop'
            )

            # Add TLX scores if available
            if self.tlx_scores:
                for dimension, score in self.tlx_scores.items():
                    final_df[f'TLX_{dimension.replace(" ", "_")}'] = score

                tlx_cols = [f'TLX_{dim.replace(" ", "_")}' for dim in self.tlx_scores.keys()]
                final_df['TLX_Overall'] = final_df[tlx_cols].mean(axis=1)

                # Combined Label
                def combined_label(row):
                    stress = int(row['Stress_Label'])
                    tlx = row['TLX_Overall']

                    if tlx > 70 and stress == 2:
                        return 2
                    elif tlx < 40 and stress == 0:
                        return 0
                    else:
                        return 1

                final_df['Combined_Stress_TLX_Label'] = final_df.apply(combined_label, axis=1)

                # Comment Stress Label
                def comment_stress_label(row):
                    physio_stress = int(row['Stress_Label'])
                    effort = row.get('TLX_Effort', 50)
                    frustration = row.get('TLX_Frustration', 50)
                    mental_load = row.get('TLX_Mental_Load', 50)
                    subjective_stress = (effort + frustration + mental_load) / 3
                    hr = row['PPG_HR']
                    hrv = row['PPG_RMSSD']

                    score = 0
                    if physio_stress == 2:
                        score += 40
                    elif physio_stress == 1:
                        score += 20

                    score += (subjective_stress / 100) * 30

                    if hr > 85:
                        score += 15
                    elif hr > 75:
                        score += 10
                    elif hr > 65:
                        score += 5

                    if hrv < 25:
                        score += 15
                    elif hrv < 35:
                        score += 10
                    elif hrv < 45:
                        score += 5

                    if score >= 70:
                        return 2
                    elif score >= 40:
                        return 1
                    else:
                        return 0

                final_df['Comment_Stress_Label'] = final_df.apply(comment_stress_label, axis=1)

            # Save dataset
            final_df.to_csv(self.dataset_file, index=False)
            print(f"✅ Dataset saved: {self.dataset_file}")

            # Generate validation report
            self._generate_validation_report(final_df, eeg_df, ppg_df, bp_df)

            # Success message
            label_counts = final_df['Stress_Label'].value_counts().to_dict()
            label_str = ", ".join([f"Label {k}: {v}" for k, v in sorted(label_counts.items())])

            success_msg = (
                f"✅ Dataset Generated Successfully!\n\n"
                f"📊 Summary:\n"
                f"• Total Windows: {len(rows)}\n"
                f"• Window Size: {WIN_SEC} seconds\n"
                f"• EEG Samples: {len(eeg_df)} ({len(eeg_df)/EEG_FS:.1f}s)\n"
                f"• PPG Samples: {len(ppg_df)} ({len(ppg_df)/PPG_FS:.1f}s)\n"
                f"• Recording Duration: {self.recording_duration}s\n"
                f"• Physiological Stress: {label_str}\n"
            )

            if self.tlx_scores:
                combined_counts = final_df['Combined_Stress_TLX_Label'].value_counts().to_dict()
                combined_str = ", ".join([f"Label {k}: {v}" for k, v in sorted(combined_counts.items())])
                comment_counts = final_df['Comment_Stress_Label'].value_counts().to_dict()
                comment_str = ", ".join([f"Label {k}: {v}" for k, v in sorted(comment_counts.items())])
                success_msg += f"• Combined TLX: {combined_str}\n"
                success_msg += f"• Comment Stress: {comment_str}\n"

            self.status_label.setText(f"✅ Dataset: {len(rows)} windows from {self.recording_duration}s recording")

            QtWidgets.QMessageBox.information(self, "✅ Success", success_msg)

        except Exception as e:
            self.status_label.setText(f"❌ Error generating dataset")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "❌ Error",
                                           f"Failed to generate dataset:\n\n{str(e)}")

    def _generate_validation_report(self, dataset_df, eeg_df, ppg_df, bp_df):
        """Generate data validation report"""
        try:
            with open(self.validation_file, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("DATA VALIDATION REPORT\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"User: {self.user_name} ({self.user_id})\n")
                f.write(f"Session Folder: {self.folder}\n\n")

                f.write("-" * 70 + "\n")
                f.write("RAW DATA STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"EEG Samples: {len(eeg_df)} ({len(eeg_df)/EEG_FS:.2f} seconds)\n")
                f.write(f"PPG Samples: {len(ppg_df)} ({len(ppg_df)/PPG_FS:.2f} seconds)\n")
                f.write(f"BP Measurements: {len(bp_df)}\n")
                f.write(f"Recording Duration: {self.recording_duration} seconds\n\n")

                f.write("-" * 70 + "\n")
                f.write("DATASET FEATURES\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total Windows: {len(dataset_df)}\n")
                f.write(f"Window Size: 5 seconds\n")
                f.write(f"Expected Windows: {self.recording_duration // 5}\n")
                f.write(f"Features per Window: {len(dataset_df.columns)}\n\n")

                f.write("Physiological Stress Label Distribution:\n")
                label_counts = dataset_df['Stress_Label'].value_counts()
                for label in ['0', '1', '2']:
                    count = label_counts.get(label, 0)
                    pct = (count / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
                    f.write(f"  Label {label}: {count} ({pct:.1f}%)\n")

                if 'Comment_Stress_Label' in dataset_df.columns:
                    f.write("\nComment Stress Label Distribution:\n")
                    comment_counts = dataset_df['Comment_Stress_Label'].value_counts()
                    for label in [0, 1, 2]:
                        count = comment_counts.get(label, 0)
                        pct = (count / len(dataset_df) * 100) if len(dataset_df) > 0 else 0
                        label_name = ["LOW STRESS", "MEDIUM STRESS", "HIGH STRESS"][label]
                        f.write(f"  Label {label} ({label_name}): {count} ({pct:.1f}%)\n")

                if self.tlx_scores:
                    f.write("\n" + "-" * 70 + "\n")
                    f.write("NASA-TLX SCORES\n")
                    f.write("-" * 70 + "\n")
                    for dim, score in self.tlx_scores.items():
                        f.write(f"{dim}: {score}/100\n")
                    overall = sum(self.tlx_scores.values()) / len(self.tlx_scores)
                    f.write(f"\nOverall TLX: {overall:.1f}/100\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("VALIDATION COMPLETE\n")
                f.write("=" * 70 + "\n")

            print(f"✅ Validation report saved: {self.validation_file}")

        except Exception as e:
            print(f"⚠️ Failed to generate validation report: {e}")

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