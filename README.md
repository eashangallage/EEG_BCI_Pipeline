# Project Documentation

This project provides a complete pipeline for Electroencephalography (EEG) data processing, including loading XDF files, preprocessing (filtering, epoching, artifact rejection, smoothing), feature extraction, and training/evaluating a Linear Discriminant Analysis (LDA) classifier for Brain-Computer Interface (BCI) applications, specifically for an RSVP (Rapid Serial Visual Presentation) paradigm. It also includes a real-time classifier leveraging Lab Streaming Layer (LSL).

This project implements an EEG processing and classification pipeline, designed to work with XDF (Extensible Data Format) files, commonly used for biosignal data. It includes utilities for data loading, preprocessing, epoching, feature extraction, and training a Linear Discriminant Analysis (LDA) classifier. A real-time classifier using Lab Streaming Layer (LSL) is also provided.

### Files Overview

* **`eeg_load_xdf.py`**:
    * **Purpose**: This script handles the loading of XDF files and structuring the data as an EEG dataset. It includes functionalities to print a summary of available streams and marker IDs within an XDF file. It can select the main data stream by name or type (with fallbacks to 'EEG' or 'ExG'). It also extracts events from marker/event streams and aligns them to the main data stream's timestamps.
    * **Key Functions**:
        * `print_summary(streams)`: Prints details about available streams and a summary of marker IDs.
        * `eeg_load_xdf(filename, ...)`: Loads EEG data from an XDF file, handling stream selection and event extraction.
        * `epoch_data(raw, target_ids, non_target_ids, ...)`: Extracts epochs around specified markers and performs baseline correction.
        * `plot_erp_comparison(targets, non_targets, ...)`: Plots Event-Related Potentials (ERPs) for target and non-target epochs for visual comparison.

* **`eeg_pipeline.py`**:
    * **Purpose**: This script contains the core EEG processing pipeline. It encompasses filtering (notch and bandpass), epoching, bad-epoch removal, smoothing, feature extraction, Z-score normalization, LDA training with balanced classes, K-fold cross-validation, and confusion matrix generation.
    * **Key Functions**:
        * `filter_signal(data, fs, hf)`: Applies bandstop and bandpass filters to the EEG data.
        * `remove_bad_epochs(epochs, labels, p2p_max)`: Rejects epochs based on a peak-to-peak amplitude threshold.
        * `smooth_epochs(epochs, window)`: Smooths each trial using a moving average.
        * `extract_features(epochs)`: Flattens epochs into a 2D feature matrix suitable for classification.
        * `run_pipeline(...)`: Executes the full EEG processing and classification pipeline, including model training and evaluation. It also saves the trained LDA model and normalization statistics to a `.pkl` file for later use in real-time classification.

* **`erp_by_image.py`**:
    * **Purpose**: This script is designed to plot average ERPs for each image ID, both before and after filtering, with baseline correction. It's useful for visualizing the effect of filtering and understanding the brain's response to different stimuli.
    * **Usage**: `python erp_by_image.py [xdf_file] --streamtype ExG --baseline 0.2 --post 1.0`

* **`main.py`**:
    * **Purpose**: This is the main entry point for training the EEG classifier. It loads an XDF file, displays a summary of its streams and markers, loads the EEG data, epochs it, plots initial ERPs, and then calls the `run_pipeline` function from `eeg_pipeline.py` to train and evaluate the LDA classifier.
    * **How the classifier is trained**: When `main.py` is called with a file path (e.g., `python main.py ../Data/rsvp_bci_recording_001_20250512_200s_100ms_1.xdf`), it performs the following steps to train the classifier:
        1.  **Load XDF and Summarize**: It first loads the specified XDF file and prints a summary of its contents, including available streams and marker IDs.
        2.  **Load EEG Data**: It then loads the raw EEG data from the selected stream (defaulting to 'ExG' or 'EEG' if not specified) and provides information about the sampling rate and number of channels.
        3.  **Initial Epoching and ERP Plotting**: The raw data is epoched based on target and non-target marker IDs (defaulting to '6' as target and others as non-targets from 1-10). An initial ERP comparison plot is shown.
        4.  **Run Full Pipeline (`run_pipeline` call)**: The `run_pipeline` function from `eeg_pipeline.py` is invoked with the XDF filename and various parameters (baseline, post-stimulus window, target/non-target IDs, filtering frequency, peak-to-peak artifact rejection threshold, K-fold splits, and smoothing window).
        5.  **Preprocessing**: Inside `run_pipeline`, the raw data is filtered (notch and bandpass), re-epoched, and bad epochs (those exceeding a `p2p_max` amplitude) are removed. The clean epochs are then smoothed.
        6.  **Feature Extraction and Normalization**: The preprocessed epochs are flattened into a 2D feature matrix, and Z-score normalization is applied to these features using their mean and standard deviation.
        7.  **LDA Training and Cross-Validation**: A Linear Discriminant Analysis (LDA) classifier is initialized with balanced priors. K-fold stratified cross-validation is performed to evaluate the model's performance on unseen data by training and predicting on different folds.
        8.  **Performance Metrics and Model Saving**: The confusion matrix, accuracy, precision, recall, and F1 score are calculated and printed. Finally, the trained LDA model, along with the normalization means and standard deviations, and the pipeline settings, are saved to a file named `eeg_online_model.pkl` using `joblib`. This saved model can then be used for real-time classification without needing to retrain.

* **`real_time_classifier.py`**:
    * **Purpose**: This script implements a real-time EEG RSVP classifier. It connects to LSL streams for ExG data and markers, buffers incoming samples, and on 'Image:X' markers, extracts epochs, preprocesses them using the saved model's parameters (filtering, smoothing, normalization), and predicts the class using the pre-trained LDA model. It tallies votes for target images and provides continuous feedback.
    * **Functionality**:
        * Loads the pre-trained LDA model and normalization statistics from `eeg_online_model.pkl`.
        * Connects to LSL 'ExG' and 'Markers' streams.
        * Main loop continuously pulls samples and markers.
        * Upon an 'Image' or 'Target' marker, an epoch is extracted from a ring buffer, filtered, smoothed, normalized, and classified by the LDA.
        * If the prediction is a target (1), the image ID receives a vote.
        * Handles 'Break:Start' to pause collection and report current votes, and 'Break:End' to resume.
        * Stops and reports the final most-voted image upon 'Experiment:End'.

* **`test_classifier.py`**:
    * **Purpose**: This script is used to evaluate a pre-trained LDA model on a *new* XDF file. It loads the saved model, processes the new data through the same pipeline steps as training (filtering, epoching, bad-epoch removal, smoothing, feature extraction, and normalization using the *saved* stats), and then makes predictions.
    * **Output**: It reports the confusion matrix, accuracy, precision, recall, and F1 score of the classifier on the new dataset, and displays the confusion matrix plot.
    * **Usage**: `python test_classifier.py [path_to_new_xdf_file]`