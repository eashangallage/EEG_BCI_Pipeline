# !/usr/bin/env python3
"""
Plot average ERPs per image ID, before and after filtering,
with baseline-correction so all traces start at zero.

Usage:
    python erp_by_image.py [xdf_file] --streamtype ExG --baseline 0.2 --post 1.0
"""
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from eeg_load_xdf import eeg_load_xdf
from eeg_pipeline import filter_signal


def main(dir,block):
    # default_dir = "E:/Develop/03_Scripts/02_python/Data/rsvp_bci_recording_001_20250512_200s_100ms_4.xdf"
    default_dir = dir
    parser = argparse.ArgumentParser(description="ERP by image before/after filter")
    parser.add_argument('xdf_file', nargs='?',
                        default=default_dir,
                        help="XDF file to analyze")
    parser.add_argument('--streamname', default=None, help="Stream name override")
    parser.add_argument('--streamtype', default='ExG', help="Stream type override")
    parser.add_argument('--baseline', type=float, default=0.2, help="Pre-stimulus baseline (s)")
    parser.add_argument('--post', type=float, default=1.0, help="Post-stimulus window (s)")
    parser.add_argument('--hf', type=float, default=35, help="Notch start (Hz)")
    args = parser.parse_args()

    # Load raw data
    raw = eeg_load_xdf(args.xdf_file,
                       streamname=args.streamname,
                       streamtype=args.streamtype)
    fs = raw['srate']
    data_raw = raw['data']  # channels x samples
    total_pts = data_raw.shape[1]

    # Filtered data copy
    data_filt = filter_signal(data_raw, fs, hf=args.hf)

    # Epoch parameters
    pre_pts = int(args.baseline * fs)
    post_pts = int(args.post * fs)
    epoch_len = pre_pts + post_pts
    times = np.linspace(-args.baseline, args.post, epoch_len, endpoint=False)

    # Collect epochs per ID
    erp_raw = {}
    erp_filt = {}
    for ev in raw['event']:
        lat = ev.get('latency')
        if lat is None:
            continue
        if lat - pre_pts < 0 or lat + post_pts > total_pts:
            continue
        # extract numeric ID
        ev_str = ev['type'][0] if isinstance(ev['type'], (list, tuple)) else ev['type']
        m = re.search(r"(\d+)", str(ev_str))
        if not m:
            continue
        id_str = m.group(1)
        # extract epochs
        epoch_r = data_raw[:, lat-pre_pts:lat+post_pts]
        epoch_f = data_filt[:, lat-pre_pts:lat+post_pts]
        # baseline-correct each epoch
        base_r = epoch_r[:, :pre_pts].mean(axis=1, keepdims=True)
        base_f = epoch_f[:, :pre_pts].mean(axis=1, keepdims=True)
        epoch_r = epoch_r - base_r
        epoch_f = epoch_f - base_f
        # store
        erp_raw.setdefault(id_str, []).append(epoch_r)
        erp_filt.setdefault(id_str, []).append(epoch_f)

    # Prepare plotting
    ids = sorted(erp_raw.keys(), key=lambda x: int(x))
    n = len(ids)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols,3*rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, id_str in enumerate(ids):
        ax = axes[idx]
        # stack and average over channels & trials
        arr_r = np.stack(erp_raw[id_str], axis=2)
        arr_f = np.stack(erp_filt[id_str], axis=2)
        avg_r = arr_r.mean(axis=(0,2))
        avg_f = arr_f.mean(axis=(0,2))
        # plot
        ax.plot(times, avg_f, 'b-', label='Filtered')
        ax.plot(times, avg_r, 'k--', label='Raw', alpha=0.6)
        ax.axvline(0, color='gray', linestyle=':')
        ax.set_title(f"Image {id_str} (n={arr_r.shape[2]})")
        if idx == 0:
            ax.legend()
        if idx % cols == 0:
            ax.set_ylabel('Amplitude')
        if idx // cols == rows-1:
            ax.set_xlabel('Time (s)')

    # remove extra axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'ERP per image ID (baseline-corrected raw vs filtered)\n {default_dir}', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show(block=block)

if __name__ == '__main__':
    _block = False
    _100ms = True
    _150ms = not _100ms

    if _100ms:
        details = "200s_100ms"
    else:
        details = "120s_150ms"

    print(_100ms,_150ms)
    for i in range(1,6):
        # dir = f"E:/Develop/03_Scripts/02_python/Data/rsvp_bci_recording_001_20250512_{details}_{i}.xdf"
        dir = f"../Data/rsvp_bci_recording_001_20250512_{details}_{i}.xdf"
        if i == 5:
            _block = True
        main(dir,_block)
