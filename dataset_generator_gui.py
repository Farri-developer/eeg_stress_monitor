import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from PySide6 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, welch, find_peaks


DEFAULT_DIR = r"D:\DataSet/farhanQ1(22-arid-3981)_20260206_200505"
MIN_GAMMA_FS = 100
WIN_SEC = 30


# ================= EEG STRESS =================
def eeg_stress_score(beta_alpha_values):
    mean_ratio = np.mean(beta_alpha_values)

    if mean_ratio < 1.2:
        return 0
    elif mean_ratio < 2.0:
        return 1
    else:
        return 2


# ================= PHYSIOLOGICAL LABEL =================
def physiological_label(hr, sdnn, rmssd, pnn50, delta_sys, eeg_score):

    stress_score = 0

    if hr > 95:
        stress_score += 1

    if rmssd < 25 or sdnn < 50:
        stress_score += 1
    elif 25 <= rmssd < 50 or 50 <= sdnn < 100:
        stress_score += 0.5

    if pnn50 < 3:
        stress_score += 1

    if abs(delta_sys) >= 5:
        stress_score += 1

    # EEG contribution (weighted)
    stress_score += eeg_score * 0.5

    if stress_score <= 1.5:
        return 0
    elif stress_score <= 3:
        return 1
    else:
        return 2


# ================= SIGNAL PROCESSING =================
def bandpass_filter(data, low, high, fs):
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def compute_band_powers(signal, fs, include_gamma=True):

    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
    }

    if include_gamma and fs >= MIN_GAMMA_FS:
        bands["Gamma"] = (30, 50)

    f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))

    powers = {}
    for band, (low, high) in bands.items():
        mask = (f >= low) & (f < high)
        powers[band] = np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else 0.0

    if "Gamma" not in powers:
        powers["Gamma"] = 0.0

    return powers


def extract_ppg_hrv(signal, fs):

    if len(signal) < fs * 2:
        return 0, 0, 0, 0

    peaks, _ = find_peaks(signal, distance=fs * 0.4)

    if len(peaks) < 2:
        return 0, 0, 0, 0

    rr = np.diff(peaks) / fs * 1000

    hr = 60 / (np.mean(rr) / 1000)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100

    return hr, sdnn, rmssd, pnn50


# ================= GUI =================
class DatasetGenerator(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Stress Dataset Generator")
        self.setFixedSize(600, 500)

        self.eeg_file = ""
        self.ppg_file = ""
        self.bp_file = ""
        self.tlx_file = ""

        self.init_ui()

    def init_ui(self):

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Multimodal Stress Dataset Generator")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        group = QtWidgets.QGroupBox("Select Input Files")
        grid = QtWidgets.QGridLayout(group)

        self.lbl_eeg = QtWidgets.QLabel("EEG not selected")
        self.lbl_ppg = QtWidgets.QLabel("PPG not selected")
        self.lbl_bp = QtWidgets.QLabel("BP not selected")
        self.lbl_tlx = QtWidgets.QLabel("TLX not selected")

        btns = [
            ("Select EEG", self.select_eeg, self.lbl_eeg),
            ("Select PPG", self.select_ppg, self.lbl_ppg),
            ("Select BP", self.select_bp, self.lbl_bp),
            ("Select TLX", self.select_tlx, self.lbl_tlx),
        ]

        for i, (text, func, label) in enumerate(btns):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(func)
            grid.addWidget(b, i, 0)
            grid.addWidget(label, i, 1)

        layout.addWidget(group)

        self.btn_generate = QtWidgets.QPushButton("Generate Final Dataset")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.generate_dataset)
        layout.addWidget(self.btn_generate)

    def update_btn(self):
        if all([self.eeg_file, self.ppg_file, self.bp_file, self.tlx_file]):
            self.btn_generate.setEnabled(True)

    def select_eeg(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "EEG CSV", DEFAULT_DIR, "CSV (*.csv)")
        if f:
            self.eeg_file = f
            self.lbl_eeg.setText(os.path.basename(f))
            self.update_btn()

    def select_ppg(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "PPG CSV", DEFAULT_DIR, "CSV (*.csv)")
        if f:
            self.ppg_file = f
            self.lbl_ppg.setText(os.path.basename(f))
            self.update_btn()

    def select_bp(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "BP CSV", DEFAULT_DIR, "CSV (*.csv)")
        if f:
            self.bp_file = f
            self.lbl_bp.setText(os.path.basename(f))
            self.update_btn()

    def select_tlx(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "TLX CSV", DEFAULT_DIR, "CSV (*.csv)")
        if f:
            self.tlx_file = f
            self.lbl_tlx.setText(os.path.basename(f))
            self.update_btn()

    # ================= DATASET =================
    def generate_dataset(self):

        try:
            eeg_df = pd.read_csv(self.eeg_file)
            ppg_df = pd.read_csv(self.ppg_file)
            bp_df = pd.read_csv(self.bp_file)
            tlx_df = pd.read_csv(self.tlx_file)

            tlx_df['Dimension'] = tlx_df['Dimension'].astype(str).str.strip()

            if any(tlx_df['Dimension'].str.lower() == 'stress_label'):
                tlx_label = int(
                    tlx_df.loc[
                        tlx_df['Dimension'].str.lower() == 'stress_label',
                        'Score'
                    ].values[0]
                )
            else:
                raise ValueError("Stress_Label not found in TLX file")

            eeg_fs = len(eeg_df) / (eeg_df['lsl_timestamp'].iloc[-1] - eeg_df['lsl_timestamp'].iloc[0])
            ppg_fs = len(ppg_df) / (ppg_df['lsl_timestamp'].iloc[-1] - ppg_df['lsl_timestamp'].iloc[0])
            include_gamma = eeg_fs >= MIN_GAMMA_FS

            question = bp_df[bp_df['label'] == 'Question-End'].iloc[-1]
            delta_sys = question['DeltaSYS']
            delta_dia = question['DeltaDIA']
            delta_pulse = question['DeltaPulse']

            rows = []

            start_time = max(eeg_df['lsl_timestamp'].iloc[0], ppg_df['lsl_timestamp'].iloc[0])
            end_time = min(eeg_df['lsl_timestamp'].iloc[-1], ppg_df['lsl_timestamp'].iloc[-1])
            num_windows = int((end_time - start_time) / WIN_SEC)

            for w in range(num_windows):

                win_start = start_time + w * WIN_SEC
                win_end = win_start + WIN_SEC

                eeg_win = eeg_df[(eeg_df['lsl_timestamp'] >= win_start) & (eeg_df['lsl_timestamp'] < win_end)]
                ppg_win = ppg_df[(ppg_df['lsl_timestamp'] >= win_start) & (ppg_df['lsl_timestamp'] < win_end)]

                if len(eeg_win) < 10 or len(ppg_win) < 10:
                    continue

                features = {
                    'Time_Start': win_start,
                    'Time_End': win_end,
                    'Window_Index': w,
                }

                eeg_vals = bandpass_filter(
                    eeg_win[['EEG1', 'EEG2', 'EEG3', 'EEG4']].values,
                    0.5, 45, eeg_fs
                )

                beta_alpha_list = []

                for ch in range(4):
                    powers = compute_band_powers(eeg_vals[:, ch], eeg_fs, include_gamma)

                    for band in powers:
                        features[f'EEG{ch+1}_{band}'] = powers[band]

                    beta_alpha = powers['Beta'] / (powers['Alpha'] + 1e-6)
                    features[f'EEG{ch+1}_BetaAlpha'] = beta_alpha
                    beta_alpha_list.append(beta_alpha)

                eeg_score = eeg_stress_score(beta_alpha_list)

                hr, sdnn, rmssd, pnn50 = extract_ppg_hrv(ppg_win['PPG1'].values, ppg_fs)

                phys_label = physiological_label(
                    hr, sdnn, rmssd, pnn50, delta_sys, eeg_score
                )

                final_label = int(round((phys_label + tlx_label) / 2))

                features.update({
                    'DeltaSYS': delta_sys,
                    'DeltaDIA': delta_dia,
                    'DeltaPULSE': delta_pulse,
                    'EEG_Score': eeg_score,
                    'TLX_Label': tlx_label,
                    'Physio_Label': phys_label,
                    'Final_Stress_Label': final_label,
                })

                rows.append(features)

            df = pd.DataFrame(rows)

            save_dir = os.path.dirname(self.eeg_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"final_dataset_{timestamp}.csv")

            df.to_csv(save_path, index=False)

            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Dataset saved at:\n{save_path}\nTotal Windows: {len(df)}"
            )

        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DatasetGenerator()
    window.show()
    sys.exit(app.exec())
