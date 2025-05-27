# !/usr/bin/env python3
"""
Import an XDF file and structure it as an EEG dataset.
Converted from MATLAB `eeg_load_xdf` by Christian Kothe, SCN, UCSD.
Default behavior with no CLI arguments: list available streams, summarize markers, then epoch and plot using default file and defaults.
Includes epoching and ERP plotting utilities with automatic fallback from EEG to ExG stream.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyxdf import load_xdf
import re

# from eeg_pipeline import run_pipeline


def print_summary(streams):
    """
    Print a summary of available streams and marker IDs.

    Parameters:
        streams (list): list of streams returned by load_xdf
    """
    import pandas as pd
    # Available streams
    print("Available streams:")
    for i, s in enumerate(streams, start=1):
        info = s['info']
        name = info.get('name')
        name = name[0] if isinstance(name, list) else name
        print(f"  {i}. {name or 'Unnamed'}")
        stype = info.get('type')
        stype = stype[0] if isinstance(stype, list) else stype
        if stype:
            print(f"      Type: {stype}")
        cc = info.get('channel_count')
        cc = cc[0] if isinstance(cc, list) else cc
        if cc:
            print(f"      Channels: {cc}")
        print(f"      Samples: {len(s['time_stamps'])}")

    # Marker summary
    entries = []
    for s in streams:
        typ = s['info'].get('type')
        typ = typ[0] if isinstance(typ, list) else typ
        if typ in ('Markers','Events'):
            mname = s['info'].get('name')
            mname = mname[0] if isinstance(mname, list) else mname
            for val in s['time_series']:
                entries.append({'type': mname, 'id': val})
                
    marker_df = pd.DataFrame(entries)
    print("Marker ID summary by type:")
    for mtype in marker_df['type'].unique():
        ids = marker_df[marker_df['type']==mtype]['id'].value_counts()
        print(f"  {mtype} IDs: {', '.join(map(str, ids.index.tolist()))}")
        # print(f"  {mtype} ID counts: {dict(ids)}")


def cart2sph(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / radius)
    return theta, phi, radius


def cart2pol(x, y):
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, radius


def eeg_load_xdf(filename,
                 streams=None,
                 header=None,
                 streamname=None,
                 streamtype=None,
                 effective_rate=False,
                 exclude_markerstreams=None):
    """
    Load EEG data from an XDF file or from preloaded streams.

    Parameters:
        filename (str): path to XDF file
        streams (list): optional preloaded streams (from load_xdf)
        header (dict): optional preloaded header
        streamname (str): specific stream name to select
        streamtype (str): preferred stream type (fallback to 'EEG', 'ExG')
        effective_rate (bool): use effective sampling rate if available
        exclude_markerstreams (list): marker stream names to skip

    Returns:
        raw (dict): contains 'data', 'srate', 'event', 'streamtype', etc.
    """
    # Load or reuse streams/header
    if exclude_markerstreams is None:
        exclude_markerstreams = []
    if streams is None or header is None:
        streams, header = load_xdf(filename)

    # 1) Select main data stream: by name or type fallback
    selected = None
    # (a) by streamname
    if streamname:
        for s in streams:
            name = s['info'].get('name')
            name = name[0] if isinstance(name, list) else name
            if name == streamname:
                selected = s
                break
        if not selected:
            raise ValueError(f"No stream named '{streamname}' found.")
    # (b) by streamtype, then default order
    else:
        candidates = ([streamtype] if streamtype else []) + ['EEG', 'ExG']
        for t in candidates:
            for s in streams:
                typ = s['info'].get('type')
                typ = typ[0] if isinstance(typ, list) else typ
                if typ == t:
                    selected = s
                    streamtype = t
                    break
            if selected:
                break
        if not selected:
            # gather available types for error message
            avail = { (s['info'].get('type')[0] if isinstance(s['info'].get('type'), list) else s['info'].get('type'))
                      for s in streams }
            raise ValueError(f"No suitable data stream found; available types: {avail}")

    # 2) Build raw dict
    raw = {}
    data_arr = np.array(selected['time_series']).T  # [channels x samples]
    raw['data'] = data_arr
    raw['nbchan'], raw['pnts'] = data_arr.shape
    raw['filepath'], raw['filename'] = os.path.split(filename)

    # 3) Sampling rate
    info = selected['info']
    nominal = info.get('nominal_srate')
    # handle list or direct string/number
    raw['srate'] = float(nominal[0]) if isinstance(nominal, list) else float(nominal)
    if effective_rate and np.isfinite(info.get('effective_srate', np.nan)):
        raw['srate'] = info['effective_srate']

    # 4) Event extraction: combine all marker/event streams
    events = []
    for s in streams:
        typ = s['info'].get('type')
        typ = typ[0] if isinstance(typ, list) else typ
        name = s['info'].get('name')
        name = name[0] if isinstance(name, list) else name
        if typ in ('Markers', 'Events') and name not in exclude_markerstreams:
            # align each timestamp to main stream indices
            main_ts = np.array(selected['time_stamps'])
            for m_val, m_ts in zip(s['time_series'], s['time_stamps']):
                latency = int(np.abs(main_ts - m_ts).argmin())
                events.append({'type': str(m_val), 'latency': latency})
    raw['event'] = events
    raw['streamtype'] = streamtype

    return raw


def epoch_data(raw, target_ids, non_target_ids, baseline_s=0.2, poststim_s=1.0):
    """
    Extracts epochs around markers *and* performs baseline correction.

    Returns:
        targets      : np.ndarray [chan x time x epochs]
        non_targets  : np.ndarray [chan x time x epochs]
    """
    srate = raw['srate']
    pre    = int(baseline_s * srate)
    post   = int(poststim_s * srate)
    length = pre + post

    data = raw['data']
    targ_epochs, nontarg_epochs = [], []

    for ev in raw['event']:
        lat = ev['latency']
        # skip if epoch would exceed data bounds
        if lat - pre < 0 or lat + post > raw['pnts']:
            continue

        # extract numeric ID from marker string
        ev_str = ev['type'][0] if isinstance(ev['type'], (list, tuple)) else ev['type']
        m = re.search(r"(\d+)", str(ev_str))
        ev_id = m.group(1) if m else str(ev_str)

        # grab the raw epoch [chan x time]
        epoch = data[:, lat-pre : lat+post].copy()

        # --- baseline correction: subtract mean of the pre-stimulus window ---
        baseline = epoch[:, :pre].mean(axis=1, keepdims=True)
        epoch = epoch - baseline

        # sort into target vs. non-target
        if ev_id in map(str, target_ids):
            targ_epochs.append(epoch)
        elif ev_id in map(str, non_target_ids):
            nontarg_epochs.append(epoch)

    # stack (or return empty arrays if no epochs)
    if targ_epochs:
        targets = np.stack(targ_epochs, axis=2)
    else:
        targets = np.empty((raw['nbchan'], length, 0))
    if nontarg_epochs:
        non_targets = np.stack(nontarg_epochs, axis=2)
    else:
        non_targets = np.empty((raw['nbchan'], length, 0))

    return targets, non_targets


def plot_erp_comparison(targets, non_targets, baseline_s, poststim_s):
    """Plot targets (blue solid) vs non-targets (red dashed) in subplots per channel."""
    nchan = targets.shape[0]
    length = targets.shape[1]
    times = np.linspace(-baseline_s, poststim_s, length)

    fig, axes = plt.subplots(nchan, 1, figsize=(8, 2*nchan), sharex=True)
    if nchan == 1:
        axes = [axes]
    for ch, ax in enumerate(axes):
        if targets.size:
            erp_t = np.mean(targets[ch, :, :], axis=1)
            ax.plot(times, erp_t, color='blue', label='Target')
        if non_targets.size:
            erp_nt = np.mean(non_targets[ch, :, :], axis=1)
            ax.plot(times, erp_nt, color='red', linestyle='--', label='Non-target')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_ylabel(f'Ch {ch+1}')
        if ch == 0:
            ax.legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show(block=False)
    # plt.show()