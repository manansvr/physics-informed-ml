import argparse
from processing import trainIndividualFiles, trainCombinedFiles


def main():
    """
    main entry point for battery capacity prediction model training.

    input:
        none (reads command-line arguments)

    output:
        none (trains models and generates visualizations based on configuration)

    function:
        parses command-line arguments to configure training mode (individual or combined),
        feature engineering options, hyperparameters, and file selection. routes to
        appropriate training function based on selected mode.
    """
    parser = argparse.ArgumentParser(description='train random forest model for battery capacity prediction')

    # mode: individual or combined
    parser.add_argument('--mode', type=str, default='individual', choices=['individual', 'combined'],
                        help='training mode: individual (train/test split per file) or combined (train on multiple, test on one)')

    # include cycle number as feature
    parser.add_argument('--include-cycle-num', action='store_true',
                        help='include cycle number as input feature')

    # test size for individual mode
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='test set size (0-1) for individual mode (default: 0.2)')

    # test file for combined mode
    parser.add_argument('--test-file', type=str, default='VAH07_r2.csv',
                        help='test file for combined mode (default: VAH07_r2.csv)')

    # hyperparameters
    parser.add_argument('--num-estimators', type=int, default=100,
                        help='number of trees in random forest (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    # file selection
    parser.add_argument('--files', type=str, nargs='+',
                        help='specific files to process (default: all files)')

    args = parser.parse_args()

    # default file list
    allFiles = ["VAH01_r2.csv", "VAH02_r2.csv", "VAH05_r2.csv", "VAH06_r2.csv",
                 "VAH07_r2.csv", "VAH09_r2.csv", "VAH10_r2.csv", "VAH11_r2.csv",
                 "VAH12_r2.csv", "VAH13_r2.csv", "VAH15_r2.csv", "VAH16_r2.csv",
                 "VAH17_r2.csv", "VAH20_r2.csv", "VAH22_r2.csv", "VAH23_r2.csv",
                 "VAH24_r2.csv", "VAH25_r2.csv", "VAH26_r2.csv", "VAH27_r2.csv",
                 "VAH28_r2.csv", "VAH30_r2.csv"]

    filePaths = args.files if args.files else allFiles

    if args.mode == 'individual':
        trainIndividualFiles(filePaths, args.include_cycle_num, args.test_size,
                              args.num_estimators, args.seed)
    else:  # combined mode
        trainCombinedFiles(filePaths, args.test_file, args.include_cycle_num,
                            args.num_estimators, args.seed)


if __name__ == "__main__":
    main()
