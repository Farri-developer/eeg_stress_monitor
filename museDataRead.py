import os, sys, csv, time, subprocess, traceback
from PySide6 import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_byprop

# ===== USER INPUT =====
session_id = input("Enter Session Number: ").strip()
name = input("Enter Your Name: ").strip()

if not session_id.isdigit():
    raise ValueError("Session must be numeric")

folder = f"Session_{session_id}_{name.replace(' ','_')}"
os.makedirs(folder, exist_ok=True)

eeg_file = f"{folder}/raw_eeg.csv"
ppg_file = f"{folder}/raw_ppg.csv"

for file, header in [
    (eeg_file, ["timestamp","ch1","ch2","ch3","ch4"]),
    (ppg_file, ["timestamp","ch1","ch2","ch3","ch4","HR"])
]:
    if not os.path.exists(file):
        with open(file,"w",newline="") as f:
            csv.writer(f).writerow(header)

# ============================================================
class MuseGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muse EEG + PPG Recorder")
        self.resize(900, 380)

        # -------- STATE --------
        self.proc = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.eeg_rec = False
        self.ppg_rec = False

        # ================= UI =================
        main = QtWidgets.QHBoxLayout()

        # ===== EEG COLUMN =====
        eeg_box = QtWidgets.QGroupBox("EEG")
        eeg_l = QtWidgets.QVBoxLayout()

        self.btn_eeg_on  = QtWidgets.QPushButton("EEG Record Start")
        self.btn_eeg_off = QtWidgets.QPushButton("EEG Record Stop")
        self.btn_eeg_on.setEnabled(False)
        self.btn_eeg_off.setEnabled(False)

        eeg_l.addWidget(self.btn_eeg_on)
        eeg_l.addWidget(self.btn_eeg_off)
        eeg_box.setLayout(eeg_l)

        # ===== PPG COLUMN =====
        ppg_box = QtWidgets.QGroupBox("PPG")
        ppg_l = QtWidgets.QVBoxLayout()

        self.btn_ppg_on  = QtWidgets.QPushButton("PPG Record Start")
        self.btn_ppg_off = QtWidgets.QPushButton("PPG Record Stop")
        self.btn_ppg_on.setEnabled(False)
        self.btn_ppg_off.setEnabled(False)

        ppg_l.addWidget(self.btn_ppg_on)
        ppg_l.addWidget(self.btn_ppg_off)
        ppg_box.setLayout(ppg_l)

        # ===== COMBINED COLUMN =====
        both_box = QtWidgets.QGroupBox("Stream & Combined Control")
        both_l = QtWidgets.QVBoxLayout()

        self.btn_stream_start = QtWidgets.QPushButton("▶ Start Stream")
        self.btn_stream_stop  = QtWidgets.QPushButton("⏹ Stop Stream")

        self.btn_both_on  = QtWidgets.QPushButton("▶ EEG + PPG Record Start")
        self.btn_both_off = QtWidgets.QPushButton("⏹ EEG + PPG Record Stop")

        self.btn_stream_stop.setEnabled(False)
        self.btn_both_on.setEnabled(False)
        self.btn_both_off.setEnabled(False)

        self.status = QtWidgets.QLabel("Idle")
        self.status.setStyleSheet("font-size:14px;color:blue")

        both_l.addWidget(self.btn_stream_start)
        both_l.addWidget(self.btn_stream_stop)
        both_l.addSpacing(10)
        both_l.addWidget(self.btn_both_on)
        both_l.addWidget(self.btn_both_off)
        both_l.addSpacing(10)
        both_l.addWidget(self.status)

        both_box.setLayout(both_l)

        main.addWidget(eeg_box)
        main.addWidget(ppg_box)
        main.addWidget(both_box)

        w = QtWidgets.QWidget()
        w.setLayout(main)
        self.setCentralWidget(w)

        # ================= SIGNALS =================
        self.btn_stream_start.clicked.connect(self.start_stream)
        self.btn_stream_stop.clicked.connect(self.stop_stream)

        self.btn_eeg_on.clicked.connect(lambda: self.toggle_eeg(True))
        self.btn_eeg_off.clicked.connect(lambda: self.toggle_eeg(False))
        self.btn_ppg_on.clicked.connect(lambda: self.toggle_ppg(True))
        self.btn_ppg_off.clicked.connect(lambda: self.toggle_ppg(False))

        self.btn_both_on.clicked.connect(self.start_both)
        self.btn_both_off.clicked.connect(self.stop_both)

        # ================= TIMERS =================
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.read_data)
        self.timer.start(10)

        self.status_timer = QtCore.QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    # ============================================================
    def start_stream(self):
        try:
            self.status.setText("Connecting Muse...")
            QtWidgets.QApplication.processEvents()

            self.proc = subprocess.Popen(
                [sys.executable, "-m", "muselsl", "stream", "--ppg"]
            )
            time.sleep(5)

            eeg = resolve_byprop("type","EEG",timeout=10)
            ppg = resolve_byprop("type","PPG",timeout=10)

            if not eeg or not ppg:
                raise RuntimeError("EEG or PPG stream not found")

            self.eeg_inlet = StreamInlet(eeg[0])
            self.ppg_inlet = StreamInlet(ppg[0])

            self.btn_eeg_on.setEnabled(True)
            self.btn_ppg_on.setEnabled(True)
            self.btn_both_on.setEnabled(True)
            self.btn_stream_stop.setEnabled(True)

            self.status.setText("Stream Connected ✅")

        except Exception as e:
            self.status.setText(f"Error: {e}")
            traceback.print_exc()

    # ============================================================
    def stop_stream(self):
        self.stop_both()
        if self.proc:
            self.proc.terminate()
            self.proc = None
        self.status.setText("Stream Stopped ❌")

        self.btn_stream_stop.setEnabled(False)
        self.btn_both_on.setEnabled(False)

    # ============================================================
    def toggle_eeg(self, state):
        self.eeg_rec = state
        self.btn_eeg_on.setEnabled(not state)
        self.btn_eeg_off.setEnabled(state)

    def toggle_ppg(self, state):
        self.ppg_rec = state
        self.btn_ppg_on.setEnabled(not state)
        self.btn_ppg_off.setEnabled(state)

    def start_both(self):
        self.toggle_eeg(True)
        self.toggle_ppg(True)
        self.btn_both_off.setEnabled(True)
        self.btn_both_on.setEnabled(False)

    def stop_both(self):
        self.toggle_eeg(False)
        self.toggle_ppg(False)
        self.btn_both_off.setEnabled(False)
        self.btn_both_on.setEnabled(True)

    # ============================================================
    def read_data(self):
        ts = int(time.time())
        try:
            if self.eeg_rec and self.eeg_inlet:
                s,_ = self.eeg_inlet.pull_sample(timeout=0.0)
                if s:
                    with open(eeg_file,"a",newline="") as f:
                        csv.writer(f).writerow([ts]+s)

            if self.ppg_rec and self.ppg_inlet:
                s,_ = self.ppg_inlet.pull_sample(timeout=0.0)
                if s:
                    with open(ppg_file,"a",newline="") as f:
                        csv.writer(f).writerow([ts]+s+[0])
        except Exception as e:
            self.status.setText(f"Runtime Error: {e}")

    # ============================================================
    def update_status(self):
        self.status.setText(
            f"Stream: {'ON' if self.proc else 'OFF'} | "
            f"EEG: {'REC' if self.eeg_rec else 'OFF'} | "
            f"PPG: {'REC' if self.ppg_rec else 'OFF'}"
        )

# ============================================================
app = QtWidgets.QApplication([])
win = MuseGUI()
win.show()
app.exec()
