# from eeg_load_xdf import eeg_load_xdf, epoch_data, plot_erp_comparison
from eeg_load_xdf import *
from eeg_pipeline import run_pipeline

def main():
    default_xdf = "../Data/rsvp_bci_recording_001_20250512_200s_100ms_1.xdf"
    parser = argparse.ArgumentParser(description='Load XDF, epoch, and plot ERPs')
    parser.add_argument('filename', nargs='?', default=default_xdf, help='Path to XDF file')
    parser.add_argument('--streamname', default=None, help='Stream name to select (optional)')
    parser.add_argument('--streamtype', default=None, help='Preferred stream type (e.g., EEG, ExG)')
    parser.add_argument('--baseline', type=float, default=0.2, help='Pre-stimulus baseline in seconds')
    parser.add_argument('--post', type=float, default=1.0, help='Post-stimulus window in seconds')
    parser.add_argument('--targets', nargs='*', default=['6'], help='List of target marker IDs')
    parser.add_argument('--nontargets', nargs='*',
                        default=[str(i) for i in range(1,11) if str(i) != '6'],
                        help='List of non-target marker IDs')
    args = parser.parse_args()

    # --- Load streams once and show summary ---
    streams, header = load_xdf(args.filename)
    print_summary(streams)

    # --- Notify user we're proceeding ---
    print(">> Now loading data stream and plotting ERPs...")

    # --- Load data using preloaded streams/header ---
    raw = eeg_load_xdf(
        filename=args.filename,
        streams=streams,
        header=header,
        streamname=args.streamname,
        streamtype=args.streamtype
    )
    print(f"Using stream: {raw['streamtype']} | SR={raw['srate']}Hz | Channels={raw['nbchan']}")

    # --- Epoch data and plot comparison ERP ---
    targets, non_targets = epoch_data(
        raw, args.targets, args.nontargets,
        baseline_s=args.baseline, poststim_s=args.post
    )

    plot_erp_comparison(targets, non_targets, args.baseline, args.post)

    # after computing raw, targets, non_targets...
    run_pipeline(
        args.filename,
        baseline_s=args.baseline, 
        poststim_s=args.post,
        targets=args.targets,
        non_targets=args.nontargets,
        hf=30,
        p2p_max=100,
        kfold_splits=5,
        # save_pipeline=True
    )

if __name__ == '__main__':
    main()