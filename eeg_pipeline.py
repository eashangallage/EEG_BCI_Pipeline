# !/usr/bin/env python3
"""
EEG processing pipeline:
- Filtering (notch + bandpass)
- Epoching
- Bad-epoch removal
- Smoothing
- Feature extraction
- Z-score normalization
- LDA training with balanced classes
- K-fold cross-validation
- Confusion matrix
- ERP comparison plotting

Call this from your main script to streamline the end-to-end flow.
"""
import numpy as np
from scipy.signal import iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
# plt.ion()  # enable interactive mode so figures don't block execution

# Import helper functions from your existing script
from eeg_load_xdf import eeg_load_xdf, epoch_data, plot_erp_comparison


def filter_signal(data, fs, hf=35):
    """
    Apply a bandstop (hf+5 to 124 Hz) then bandpass (1–30 Hz) filter to the data.

    data: np.ndarray [channels x samples]
    fs: sampling rate
    hf: high-frequency interference start (Hz)
    """
    # Notch filter design (Butterworth bandstop)
    b_notch, a_notch = iirfilter(
        2, [hf + 5, 124], btype='bandstop', ftype='butter', fs=fs
    )
    data_notch = filtfilt(b_notch, a_notch, data, axis=-1)

    # Bandpass filter design (Butterworth bandpass)
    b_bp, a_bp = iirfilter(
        4, [1.0, 30.0], btype='bandpass', ftype='butter', fs=fs
    )
    data_bp = filtfilt(b_bp, a_bp, data_notch, axis=-1)
    return data_bp


def remove_bad_epochs(epochs, labels, p2p_max):
    """
    Reject epochs whose peak-to-peak exceeds p2p_max.

    Returns cleaned_epochs [chan x time x epochs], cleaned_labels, kept_indices.
    """
    good = []
    for idx in range(epochs.shape[2]):
        if np.ptp(epochs[:, :, idx]) <= p2p_max:
            good.append(idx)
    return epochs[:, :, good], np.array(labels)[good], good


def smooth_epochs(epochs, window=50):
    """
    Smooth each trial with a moving average of length `window` samples.

    epochs: [chan x time x trials]
    """
    # convolution along time axis
    kernel = np.ones(window) / window
    smoothed = np.empty_like(epochs)
    for t in range(epochs.shape[2]):
        smoothed[:, :, t] = np.array([
            np.convolve(epochs[ch, :, t], kernel, mode='same')
            for ch in range(epochs.shape[0])
        ])
    return smoothed


def extract_features(epochs):
    """
    Flatten epochs into 2D feature matrix [trials x (chan*time)].
    """
    ch, length, trials = epochs.shape
    return epochs.reshape(ch * length, trials).T


def run_pipeline(xdf_file,
                 baseline_s=0.2,
                 poststim_s=1.0,
                 targets=['6'],
                 non_targets=None,
                 hf=35,
                 p2p_max=100,
                 kfold_splits=5,
                 smoothing_win=25,
                 save_pipeline = False):
    """
    Execute the full EEG pipeline:
      1) Load & filter raw data
      2) Epoch around markers
      3) Initial ERP comparison
      4) Remove bad epochs
      5) Smooth epochs
      5a) ERP comparison after cleaning & smoothing
      6) Extract features & normalize
      7) Train and cross-validate LDA (balanced)
      8) Display confusion matrix
    """
    # Default non-targets if not provided
    if non_targets is None:
        non_targets = [str(i) for i in range(1, 11) if str(i) not in targets]

    # 1) Load raw and apply filtering
    raw = eeg_load_xdf(xdf_file)
    fs = raw['srate']
    raw['data'] = filter_signal(raw['data'], fs, hf=hf)

    # 2) Epoch around markers
    targ_epochs, nt_epochs = epoch_data(
        raw, target_ids=targets, non_target_ids=non_targets,
        baseline_s=baseline_s, poststim_s=poststim_s
    )

    # 3) ERP comparison (pre-clean)
    plot_erp_comparison(targ_epochs, nt_epochs, baseline_s, poststim_s)
    plt.pause(0.5)

    # 4) Combine epochs and perform bad-epoch removal
    labels = [1] * targ_epochs.shape[2] + [0] * nt_epochs.shape[2]
    all_epochs = np.concatenate([targ_epochs, nt_epochs], axis=2)
    clean_epochs, clean_labels, kept = remove_bad_epochs(all_epochs, labels, p2p_max)

    # 5) Smooth cleaned epochs
    smooth = smooth_epochs(clean_epochs, window=smoothing_win)

    # 5a) ERP comparison (post-clean & smoothing)
    # Split back into targets and non-targets based on clean_labels
    clean_labels = np.array(clean_labels)
    cleaned_targs = smooth[:, :, clean_labels == 1]
    cleaned_non =  smooth[:, :, clean_labels == 0]
    plot_erp_comparison(cleaned_targs, cleaned_non, baseline_s, poststim_s)
    plt.pause(0.5)

    # 6) Feature extraction & z-score normalization
    X = extract_features(smooth)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = clean_labels

    # 7) Train on full data first (optional—CV will retrain internally anyway)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])
    lda.fit(X, y)

    # 8) Then cross-validate (predicts by re‑training on each fold)
    cv = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
    y_pred = cross_val_predict(lda, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:", cm)
    plt.figure()
    plt.matshow(cm, fignum=False)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.draw(); plt.pause(0.5)

    # 9) Evaluation metrics
    acc = accuracy_score(y, y_pred)*100
    prec = precision_score(y, y_pred)*100
    rec = recall_score(y, y_pred)*100
    f1 = f1_score(y, y_pred)*100
    print(f"Accuracy: {acc:.2f} %, Precision: {prec:.2f} %, Recall: {rec:.2f} %, F1: {f1:.2f} %")
    
    # 10) Train LDA on the full (cleaned & normalized) dataset so .predict() will work later
    lda.fit(X, y)

    import joblib
    # compute & store normalization stats
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)

    # package everything into one dict
    model_bundle = {
        'lda'      : lda,
        'mean'     : X_mean,
        'std'      : X_std,
        # optional: save your pipeline settings too
        'fs'             : fs,
        'baseline_s'     : baseline_s,
        'poststim_s'     : poststim_s,
        'hf'             : hf,
        'p2p_max'        : p2p_max,
        'smoothing_win'  : smoothing_win,           # if you’re smoothing online
    }

    if save_pipeline:
        # dump to disk
        joblib.dump(model_bundle, 'eeg_online_model.pkl')
    
    plt.show()