# !/usr/bin/env python3
"""
Evaluate a pre-trained EEG LDA model on a new XDF file.

- Loads model bundle from 'eeg_online_model.pkl'.
- Loads a new XDF via pyxdf and eeg_load_xdf utilities.
- Applies filtering, epoching, bad-epoch removal, smoothing.
- Extracts features and z-score normalizes using saved stats.
- Runs the model's LDA to predict on all epochs.
- Reports accuracy, precision, recall, F1 score, and confusion matrix.
"""
import argparse
import numpy as np
from joblib import load
from pyxdf import load_xdf
from eeg_load_xdf import eeg_load_xdf, epoch_data
from eeg_pipeline import filter_signal, smooth_epochs
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt


def main():
    # default XDF file if none provided
    # default_xdf = "E:/Develop/03_Scripts/02_python/Data/online_rsvp_bci_data_00110.xdf"
    default_xdf = "../Data/rsvp_bci_recording_001_20250512_120s_150ms_2.xdf"
    parser = argparse.ArgumentParser(description='Evaluate EEG LDA model on a new XDF file')
    parser = argparse.ArgumentParser(description='Evaluate EEG LDA model on a new XDF file')
    parser.add_argument('xdf_file', nargs='?', default=default_xdf,
                        help='Path to the XDF file to evaluate (default: %(default)s)')
    parser.add_argument('--streamname', default=None, help='Stream name override')
    parser.add_argument('--streamtype', default=None, help='Stream type override (e.g., ExG)')
    parser.add_argument('--targets', nargs='+', default=['6'], help='List of target marker IDs')
    parser.add_argument('--nontargets', nargs='+',
                        default=[str(i) for i in range(1, 11) if str(i) not in ['6']],
                        help='List of non-target marker IDs')
    args = parser.parse_args()

    # Load model bundle
    bundle = load('eeg_online_model.pkl')
    lda        = bundle['lda']
    X_mean     = bundle['mean']
    X_std      = bundle['std']
    fs         = bundle['fs']
    baseline_s = bundle['baseline_s']
    poststim_s = bundle['poststim_s']
    hf         = bundle['hf']
    p2p_max    = bundle['p2p_max']
    smoothing_win = bundle.get('smoothing_win', 50)

    # 1) Load raw via eeg_load_xdf
    raw = eeg_load_xdf(
        args.xdf_file,
        streamname=args.streamname,
        streamtype=args.streamtype
    )

    # 2) Filter raw data
    raw['data'] = filter_signal(raw['data'], fs, hf)

    # 3) Epoch around markers
    targ_epochs, nt_epochs = epoch_data(
        raw,
        target_ids=args.targets,
        non_target_ids=args.nontargets,
        baseline_s=baseline_s,
        poststim_s=poststim_s
    )

    # 4) Combine and remove bad epochs
    labels = [1]*targ_epochs.shape[2] + [0]*nt_epochs.shape[2]
    all_epochs = np.concatenate([targ_epochs, nt_epochs], axis=2)
    # Bad-epoch removal
    good_idx = [i for i in range(all_epochs.shape[2]) if np.ptp(all_epochs[:, :, i]) <= p2p_max]
    clean_epochs = all_epochs[:, :, good_idx]
    clean_labels = np.array(labels)[good_idx]

    # 5) Smooth epochs
    smooth = smooth_epochs(clean_epochs, window=smoothing_win)

    # 6) Extract features and normalize
    ch, length, trials = smooth.shape
    X = smooth.reshape(ch*length, trials).T
    X = (X - X_mean) / X_std
    y = clean_labels

    # 7) Predict
    y_pred = lda.predict(X)

    # 8) Metrics
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Print results
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")

    # Plot confusion matrix
    plt.figure()
    plt.matshow(cm, fignum=False)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    main()
