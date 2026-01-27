import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time

# ===============================
FS = 256  # Muse sampling rate
WIN_SEC = 5
WIN_SAMPLES = FS * WIN_SEC
SAVE_DIR = r"D:\DatasatCombine"
os.makedirs(SAVE_DIR, exist_ok=True)

muse_path = ""
bp_path = ""


# ===============================
def bandpass(data, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=0)


# ===============================
def select_muse():
    global muse_path
    muse_path = filedialog.askopenfilename(
        title="Select Muse CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    muse_label.config(text=os.path.basename(muse_path))


def select_bp():
    global bp_path
    bp_path = filedialog.askopenfilename(
        title="Select BP CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    bp_label.config(text=os.path.basename(bp_path))


# ===============================
def extract_ppg_hrv(ppg_signal, fs=FS):
    # ppg_signal: 1D numpy array
    peaks, _ = find_peaks(ppg_signal, distance=fs * 0.5)  # minimum 0.5s between beats
    if len(peaks) < 2:
        return 0, 0, 0, 0
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    hr = 60 / (np.mean(rr_intervals) / 1000)  # bpm
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
    return hr, sdnn, rmssd, pnn50


# ===============================
def run_extraction():
    if not muse_path or not bp_path:
        messagebox.showerror("Error", "Please select both Muse and BP CSV files")
        return

    try:
        # ---------- LOAD DATA ----------
        muse = pd.read_csv(muse_path).dropna()
        muse.iloc[:, 1:] = muse.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        muse = muse.dropna()

        EEG = muse[['EEG1', 'EEG2', 'EEG3', 'EEG4']].values
        PPG = muse[['PPG2', 'PPG3', 'PPG4']].values  # Ignore PPG1 column

        EEG = bandpass(EEG, 0.5, 45, FS)
        PPG = bandpass(PPG, 0.5, 8, FS)

        # ---------- LOAD BP ----------
        bp = pd.read_csv(bp_path)
        baseline = bp[bp['label'] == 'Baseline'].iloc[0]
        question = bp[bp['label'] != 'Baseline'].iloc[-1]

        bp_features = {
            'DeltaSYS': question['SYS'] - baseline['SYS'],
            'DeltaDIA': question['DIA'] - baseline['DIA'],
            'DeltaPULSE': question['PULSE'] - baseline['PULSE']
        }

        # ---------- FEATURE EXTRACTION ----------
        rows = []
        num_windows = len(EEG) // WIN_SAMPLES

        for w in range(num_windows):
            s = w * WIN_SAMPLES
            e = s + WIN_SAMPLES

            eeg_win = EEG[s:e, :]
            ppg_win = PPG[s:e, :]

            features = {'Window': w}

            # EEG features
            for ch in range(4):
                f, pxx = welch(eeg_win[:, ch], fs=FS, nperseg=FS * 2)
                delta = np.trapz(pxx[(f >= 0.5) & (f < 4)])
                theta = np.trapz(pxx[(f >= 4) & (f < 8)])
                alpha = np.trapz(pxx[(f >= 8) & (f < 13)])
                beta = np.trapz(pxx[(f >= 13) & (f < 30)])
                features[f'EEG{ch + 1}_Delta'] = delta
                features[f'EEG{ch + 1}_Theta'] = theta
                features[f'EEG{ch + 1}_Alpha'] = alpha
                features[f'EEG{ch + 1}_Beta'] = beta
                features[f'EEG{ch + 1}_BetaAlpha'] = beta / (alpha + 1e-6)

            # PPG features: mean/std per window across all 3 channels
            features['PPG_Mean'] = np.mean(ppg_win)
            features['PPG_STD'] = np.std(ppg_win)

            # HRV per PPG channel
            hr_all, sdnn_all, rmssd_all, pnn50_all = [], [], [], []
            for ch in range(ppg_win.shape[1]):
                hr, sdnn, rmssd, pnn50 = extract_ppg_hrv(ppg_win[:, ch])
                hr_all.append(hr)
                sdnn_all.append(sdnn)
                rmssd_all.append(rmssd)
                pnn50_all.append(pnn50)
            # Average across 3 PPG channels
            features['PPG_HR'] = np.mean(hr_all)
            features['PPG_SDNN'] = np.mean(sdnn_all)
            features['PPG_RMSSD'] = np.mean(rmssd_all)
            features['PPG_pNN50'] = np.mean(pnn50_all)

            # BP deltas
            features.update(bp_features)

            rows.append(features)

        final_df = pd.DataFrame(rows)

        # ---------- SAVE FILE ----------
        timestamp = int(time.time())
        save_path = os.path.join(SAVE_DIR, f"EEG_PPG_BP_HRV_5sec_{timestamp}.csv")
        final_df.to_csv(save_path, index=False)

        messagebox.showinfo("Success", f"Dataset created!\nSaved at:\n{save_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ===============================
# GUI
root = tk.Tk()
root.title("EEG + PPG + BP + HRV Feature Extractor")
root.geometry("700x400")

tk.Label(root, text="Muse CSV (EEG+PPG)", font=("Arial", 11)).pack(pady=5)
tk.Button(root, text="Select Muse File", command=select_muse).pack()
muse_label = tk.Label(root, text="No file selected", fg="gray")
muse_label.pack()

tk.Label(root, text="BP CSV", font=("Arial", 11)).pack(pady=10)
tk.Button(root, text="Select BP File", command=select_bp).pack()
bp_label = tk.Label(root, text="No file selected", fg="gray")
bp_label.pack()

tk.Button(root, text="Run Feature Extraction", command=run_extraction,
          bg="green", fg="white", height=2, width=35).pack(pady=30)

root.mainloop()
