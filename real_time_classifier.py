# !/usr/bin/env python3
"""
Real-time EEG RSVP classifier using LSL and a pre-trained LDA model.

- Connects to ExG stream ("Explore_1C33_ExG") and marker stream ("Explore_1C33_Marker").
- Buffers incoming samples in a ring buffer for epoch extraction.
- On each 'Image:X' marker (when collection enabled), extracts the last epoch, filters, z-scores, and predicts with the LDA.
- Tallies votes for each image ID predicted as target.
- Pauses collection on 'Break:Start', resumes on 'Break:End'.
- Stops the program only when it sees 'Experiment:End', then outputs the most-voted image.
"""
import time
from collections import deque, Counter
import numpy as np
from joblib import load
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import iirfilter, filtfilt

from eeg_pipeline import filter_signal, smooth_epochs

# Load pre-trained model bundle
bundle = load('eeg_online_model.pkl')
lda        = bundle['lda']
X_mean     = bundle['mean']
X_std      = bundle['std']
fs         = bundle['fs']
baseline_s = bundle['baseline_s']
poststim_s = bundle['poststim_s']
hf         = bundle['hf']
p2p_max    = bundle['p2p_max']
smoothing_win = bundle['smoothing_win']

# Derived parameters
pre_samples  = int(baseline_s * fs)
post_samples = int(poststim_s * fs)
epoch_len    = pre_samples + post_samples

# Ring buffer for ExG samples
buffer = deque(maxlen=int(fs * 10))  # holds last 10 seconds of data  # holds last 10 seconds of data

# Voting tally
votes = Counter()

init_timeout = 100

# Connect to LSL streams
print("Resolving streams...")
marker_streams = resolve_byprop('type', 'Markers', timeout=init_timeout)
exg_streams    = resolve_byprop('type', 'ExG', timeout=init_timeout)
if not marker_streams or not exg_streams:
    raise RuntimeError("Could not find required LSL streams.")
marker_inlet = StreamInlet(marker_streams[0])
exg_inlet    = StreamInlet(exg_streams[0])
print("Streams connected. Starting real-time classification...")

# Flags
collection_enabled = True   # True when collecting Image epochs
running = True             # True until Experiment:End

print("Entering main loop...")
while running:
    # Pull ExG sample
    sample, _ = exg_inlet.pull_sample(timeout=0.0)
    if sample:
        buffer.append(sample)

    # Pull marker
    marker, _ = marker_inlet.pull_sample(timeout=0.0)
    if marker:
        m = marker[0]
        if isinstance(m, str) and ':' in m:
            label, id_str = m.split(':', 1)
            label = label.strip()
            id_str = id_str.strip()
        else:
            continue

        if (label == 'Image' or label == 'Target') and collection_enabled:
            # Ensure buffer has enough samples
            if len(buffer) < epoch_len:
                continue
            data = np.array(buffer)[-epoch_len:, :].T  # [chan x samples]
            # 1) Filter
            epoch_f = filter_signal(data, fs, hf)
            # 2) Bad-epoch removal: skip noisy
            if np.ptp(epoch_f) > p2p_max:
                continue
            # 3) Smooth
            epoch_sm = smooth_epochs(epoch_f[:, :, np.newaxis], window=smoothing_win)[:, :, 0]
            # 4) Flatten & normalize
            X = epoch_sm.flatten()
            X = (X - X_mean) / X_std
            # 5) Predict
            pred = lda.predict(X.reshape(1, -1))[0]
            if pred == 1:
                votes[id_str] += 1

        elif label == 'Break':
            if id_str == 'Start' and collection_enabled:
                # End of a collection cycle: report and reset
                if votes:
                    chosen, count = votes.most_common(1)[0]
                    total_votes = sum(votes.values())

                    print(f"Cycle predicted target image: {chosen} (votes: {count}/{total_votes})")
                else:
                    print("Cycle: no votes collected.")
                votes.clear()
                buffer.clear()
                collection_enabled = False
                print("Break start detected: pausing collection.")
            elif id_str == 'End' and not collection_enabled:
                collection_enabled = True
                print("Break end detected: resuming collection.")

        elif label == 'Experiment' and id_str == 'End':
            # Final cycle prediction
            print("Experiment end detected: final cycle prediction:")
            if votes:
                chosen, count = votes.most_common(1)[0]
                total_votes = sum(votes.values())

                print(f"Cycle predicted target image: {chosen} (votes: {count}/{total_votes})")
            else:
                print("Cycle: no votes collected.")
            running = False
    time.sleep(0.001)

print("Real-time classification completed.")