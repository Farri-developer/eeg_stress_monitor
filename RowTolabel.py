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

    # ---------- ADD LABEL HERE ----------
    # Default label = -1 (unknown)
    # Change this to 0/1/2 for Low/Medium/High stress if you have mapping
    features['Stress_Label'] = -1

    rows.append(features)
