import sys, os, csv, time, subprocess, asyncio, signal
from datetime import datetime
from PySide6 import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_byprop
from bleak import BleakClient

# ================= BASE PATH =================
BASE_DIR = r"D:\DataSet"
os.makedirs(BASE_DIR, exist_ok=True)

# ================= BP DEVICE =================
BP_ADDRESS = "18:7A:93:12:26:AE"
BP_UUID = "00002a35-0000-1000-8000-00805f9b34fb"

# ================= BP DECODER =================
def decode_bp(data):
    flags = data[0]
    systolic  = int.from_bytes(data[1:3], "little")
    diastolic = int.from_bytes(data[3:5], "little")
    mean_art  = int.from_bytes(data[5:7], "little")

    idx = 7
    if flags & 0x02:  # timestamp present
        idx += 7

    pulse = None
    if flags & 0x04:
        pulse = int.from_bytes(data[idx:idx+2], "little")

    # Fix zero MAP
    if mean_art == 0:
        mean_art = round(diastolic + (systolic - diastolic) / 3, 1)

    return systolic, diastolic, mean_art, pulse

# ================= BP THREAD =================
class BPWorker(QtCore.QThread):
    reading = QtCore.Signal(int, int, float, object)
    status = QtCore.Signal(str)

    def run(self):
        asyncio.run(self.task())

    async def task(self):
        try:
            self.status.emit("BP: Connecting… Press BP button")
            async with BleakClient(BP_ADDRESS, timeout=12) as client:
                got = False
                def handler(_, data):
                    nonlocal got
                    if not got:
                        got = True
                        self.reading.emit(*decode_bp(data))

                await client.start_notify(BP_UUID, handler)
                # Wait max 45s for reading
                for _ in range(45):
                    if got:
                        break
                    await asyncio.sleep(1)
                await client.stop_notify(BP_UUID)
                self.status.emit("BP: Disconnected")
        except Exception as e:
            self.status.emit(f"BP Error: {e}")

# ================= GUI =================
class HealthGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG + PPG + BP Monitor")
        self.resize(900, 520)

        # ===== SESSION =====
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(BASE_DIR, f"Session_{self.session}")
        os.makedirs(self.folder, exist_ok=True)
        self.eeg_file = f"{self.folder}/eeg.csv"
        self.ppg_file = f"{self.folder}/ppg.csv"
        self.bp_file = f"{self.folder}/bp.csv"
        self.init_csv()

        # ===== STATE =====
        self.proc = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.recording = False   # single state for EEG+PPG
        self.bp_count = 0

        # ===== UI =====
        layout = QtWidgets.QVBoxLayout()
        self.btn_stream_on = QtWidgets.QPushButton("▶ START MUSE STREAM")
        self.btn_stream_off = QtWidgets.QPushButton("⏹ STOP MUSE STREAM")
        self.btn_record = QtWidgets.QPushButton("▶ RECORD EEG+PPG")
        self.btn_bp = QtWidgets.QPushButton("🩺 START BP")
        self.status_stream = QtWidgets.QLabel()
        self.status_record = QtWidgets.QLabel()
        self.status_bp = QtWidgets.QLabel()
        for w in [self.btn_stream_on, self.btn_stream_off, self.btn_record, self.btn_bp,
                  self.status_stream, self.status_record, self.status_bp]:
            layout.addWidget(w)

        c = QtWidgets.QWidget()
        c.setLayout(layout)
        self.setCentralWidget(c)

        # ===== SIGNALS =====
        self.btn_stream_on.clicked.connect(self.start_stream)
        self.btn_stream_off.clicked.connect(self.stop_stream)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_bp.clicked.connect(self.start_bp)

        # ===== TIMER =====
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.read_data)
        self.timer.start(10)
        self.update_status()

    # ================= CSV =================
    def init_csv(self):
        with open(self.eeg_file, "w", newline="") as f:
            csv.writer(f).writerow(["time", "ch1", "ch2", "ch3", "ch4"])
        with open(self.ppg_file, "w", newline="") as f:
            csv.writer(f).writerow(["time", "ch1", "ch2", "ch3", "ch4"])
        with open(self.bp_file, "w", newline="") as f:
            csv.writer(f).writerow(["time", "SYS", "DIA", "MAP", "PULSE"])

    # ================= STREAM =================
    def start_stream(self):
        if self.proc:
            return
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "muselsl", "stream", "--ppg"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        time.sleep(6)  # give Muse time to start
        self.eeg_inlet = StreamInlet(resolve_byprop("type", "EEG")[0])
        self.ppg_inlet = StreamInlet(resolve_byprop("type", "PPG")[0])
        self.update_status()

    def stop_stream(self):
        if self.proc:
            os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            self.proc = None
            self.eeg_inlet = None
            self.ppg_inlet = None
            self.recording = False
            self.update_status()

    # ================= RECORD =================
    def toggle_recording(self):
        self.recording = not self.recording
        self.update_status()

    # ================= DATA =================
    def read_data(self):
        t = int(time.time())
        if self.recording and self.eeg_inlet:
            s,_ = self.eeg_inlet.pull_sample(timeout=0.0)
            if s:
                with open(self.eeg_file, "a", newline="") as f:
                    csv.writer(f).writerow([t] + s)
        if self.recording and self.ppg_inlet:
            s,_ = self.ppg_inlet.pull_sample(timeout=0.0)
            if s:
                with open(self.ppg_file, "a", newline="") as f:
                    csv.writer(f).writerow([t] + s)

    # ================= BP =================
    def start_bp(self):
        self.bp_thread = BPWorker()
        self.bp_thread.reading.connect(self.save_bp)
        self.bp_thread.status.connect(self.status_bp.setText)
        self.bp_thread.start()

    def save_bp(self, sys, dia, map_, pulse):
        self.bp_count += 1
        t = datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file, "a", newline="") as f:
            csv.writer(f).writerow([t, sys, dia, map_, pulse])
        self.status_bp.setText(f"BP Saved #{self.bp_count} ✔ SYS:{sys} DIA:{dia} P:{pulse}")

    # ================= STATUS =================
    def update_status(self):
        self.status_stream.setText(f"Muse Stream: {'ON' if self.proc else 'OFF'}")
        self.status_record.setText(f"EEG+PPG: {'REC' if self.recording else 'OFF'}")

# ================= RUN =================
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = HealthGUI()
    win.show()
    app.exec()
