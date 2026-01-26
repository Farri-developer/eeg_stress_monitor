import sys, os, csv, time, asyncio
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
    if flags & 0x02: idx += 7
    pulse = int.from_bytes(data[idx:idx+2], "little") if flags & 0x04 else None
    if mean_art == 0:
        mean_art = round(diastolic + (systolic - diastolic)/3,1)
    return systolic, diastolic, mean_art, pulse

# ================= BP WORKER =================
class BPWorker(QtCore.QThread):
    reading = QtCore.Signal(int,int,float,object)
    status = QtCore.Signal(str)
    async def task(self):
        try:
            self.status.emit("BP: Connecting… Press BP device button")
            async with BleakClient(BP_ADDRESS, timeout=12) as client:
                got = False
                def handler(_, data):
                    nonlocal got
                    if not got:
                        got=True
                        self.reading.emit(*decode_bp(data))
                await client.start_notify(BP_UUID, handler)
                for _ in range(45):
                    if got: break
                    await asyncio.sleep(1)
                await client.stop_notify(BP_UUID)
                self.status.emit("BP: Disconnected")
        except Exception as e:
            self.status.emit(f"BP Error: {e}")
    def run(self):
        asyncio.run(self.task())

# ================= GUI =================
class HealthGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG+PPG+BP Recorder")
        self.resize(750,550)

        # ===== SESSION =====
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(BASE_DIR,f"Question_{self.session}")
        os.makedirs(self.folder, exist_ok=True)

        # ===== FILES =====
        self.eeg_file = os.path.join(self.folder,"eeg_ppg_raw.csv")
        self.bp_file  = os.path.join(self.folder,"bp_raw.csv")
        with open(self.eeg_file,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["time","EEG1","EEG2","EEG3","EEG4","PPG1","PPG2","PPG3","PPG4"])
        with open(self.bp_file,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["time","label","SYS","DIA","MAP","PULSE","DeltaSYS","DeltaDIA","DeltaPulse"])

        # ===== STATE =====
        self.recording = False
        self.base_bp = None
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.proc = None

        # ===== UI =====
        layout = QtWidgets.QVBoxLayout()
        self.btn_baseline     = QtWidgets.QPushButton("🩺 Baseline BP")
        self.btn_stream_start = QtWidgets.QPushButton("▶ Start Stream")
        self.btn_question     = QtWidgets.QPushButton("▶ Start Question Recording")
        self.btn_stop         = QtWidgets.QPushButton("⏹ Stop Recording")
        self.btn_stream_stop  = QtWidgets.QPushButton("⏹ Stop Stream")
        self.btn_qend_bp      = QtWidgets.QPushButton("💓 Question-End BP")
        self.btn_collect_all  = QtWidgets.QPushButton("📁 Collect All Data")
        self.status_label     = QtWidgets.QLabel("Status: Idle")
        self.bp_status_label  = QtWidgets.QLabel("BP: Not measured")

        for w in [self.btn_baseline,self.btn_stream_start,self.btn_question,
                  self.btn_stop,self.btn_stream_stop,self.btn_qend_bp,
                  self.btn_collect_all,self.status_label,self.bp_status_label]:
            layout.addWidget(w)
        c=QtWidgets.QWidget()
        c.setLayout(layout)
        self.setCentralWidget(c)

        # ===== SIGNALS =====
        self.btn_baseline.clicked.connect(self.get_baseline)
        self.btn_stream_start.clicked.connect(self.start_stream)
        self.btn_stream_stop.clicked.connect(self.stop_stream)
        self.btn_question.clicked.connect(self.start_question)
        self.btn_stop.clicked.connect(self.stop_question)
        self.btn_qend_bp.clicked.connect(self.question_end_bp)
        self.btn_collect_all.clicked.connect(self.collect_all_data)

        # ===== TIMER =====
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.record_stream)
        self.timer.start(10)

    # ================= STREAM =================
    def start_stream(self):
        if self.proc:
            self.status_label.setText("Stream already running")
            return
        import subprocess, sys, time
        self.proc=subprocess.Popen([sys.executable,"-m","muselsl","stream","--ppg"],
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        time.sleep(6)
        self.eeg_inlet=StreamInlet(resolve_byprop("type","EEG")[0])
        self.ppg_inlet=StreamInlet(resolve_byprop("type","PPG")[0])
        self.status_label.setText("Stream started ✅")

    def stop_stream(self):
        if self.proc:
            import signal, os
            os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            self.proc = None
            self.eeg_inlet = None
            self.ppg_inlet = None
            self.status_label.setText("Stream stopped ⏹ Disconnected")
        else:
            self.status_label.setText("Stream not running")

    # ================= RECORD STREAM =================
    def record_stream(self):
        if self.recording and self.eeg_inlet and self.ppg_inlet:
            t = int(time.time())
            eeg_sample, _ = self.eeg_inlet.pull_sample(timeout=0.0)
            ppg_sample, _ = self.ppg_inlet.pull_sample(timeout=0.0)
            if eeg_sample and ppg_sample:
                row = [t] + eeg_sample + ppg_sample
                with open(self.eeg_file,"a",newline="",encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

    # ================= BASELINE BP =================
    def get_baseline(self):
        self.status_label.setText("Taking baseline BP…")
        self.bp_thread=BPWorker()
        self.bp_thread.reading.connect(self.save_baseline)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def save_baseline(self,sys,dia,map_,pulse):
        self.base_bp=(sys,dia,pulse)
        t=datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([t,"Baseline",sys,dia,map_,pulse,0,0,0])
        self.bp_status_label.setText(f"Baseline BP: SYS {sys} DIA {dia} P {pulse}")
        self.status_label.setText("Baseline BP saved ✔")

    # ================= QUESTION =================
    def start_question(self):
        self.status_label.setText(f"Recording Question…")
        self.recording=True

    # ================= STOP QUESTION =================
    def stop_question(self):
        self.recording=False
        self.status_label.setText("Recording stopped. Take Question-End BP now.")

    # ================= QUESTION-END BP =================
    def question_end_bp(self):
        if not self.base_bp:
            self.status_label.setText("Error: Baseline BP not measured!")
            return
        self.status_label.setText("Measuring Question-End BP…")
        self.bp_thread=BPWorker()
        self.bp_thread.reading.connect(self.save_question_bp)
        self.bp_thread.status.connect(self.bp_status_label.setText)
        self.bp_thread.start()

    def save_question_bp(self,sys,dia,map_,pulse):
        delta_sys = sys - self.base_bp[0]
        delta_dia = dia - self.base_bp[1]
        delta_pulse = pulse - self.base_bp[2]
        t=datetime.now().strftime("%H:%M:%S")
        with open(self.bp_file,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([t,"Question-End",sys,dia,map_,pulse,delta_sys,delta_dia,delta_pulse])
        self.bp_status_label.setText(f"Question-End BP: SYS {sys} DIA {dia} P {pulse}")
        self.status_label.setText(f"Question-End BP saved ✔ DeltaSYS:{delta_sys} DeltaDIA:{delta_dia} DeltaP:{delta_pulse}")

    # ================= COLLECT ALL DATA =================
    def collect_all_data(self):
        # Stop EEG/PPG stream if running
        if self.recording:
            self.recording=False
        if self.proc:
            self.stop_stream()

        # Take Question-End BP if baseline exists
        if self.base_bp:
            self.question_end_bp()
            QtCore.QTimer.singleShot(12000, lambda: self.status_label.setText("All data collected ✅"))
        else:
            self.status_label.setText("All data collected ✅ (No baseline BP)")

# ================= RUN =================
if __name__=="__main__":
    app=QtWidgets.QApplication([])
    win=HealthGUI()
    win.show()
    app.exec()
