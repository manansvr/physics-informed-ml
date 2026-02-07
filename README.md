# Physics-Informed Machine Learning
## Battery Degradation & Performance Prediction

This project implements physics-informed machine learning models to predict lithium-ion battery degradation and performance metrics, including Remaining Useful Life (RUL), State of Charge (SoC), and State of Health (SoH).

The project uses comprehensive battery cycling data from 22 high-density pouch cells, encompassing over 21,000 charge-discharge cycles under varied operational conditions.

**Data Features:**
- **Time**: timestamp of the measurement
- **Voltage**: battery voltage (V) - critical for SoC estimation
- **Current**: battery current (A) - key for power and thermal modeling
- **Temperature**: operating temperature (°C) - affects degradation patterns
- **Cycle Number**: charge/discharge cycle count - essential for RUL prediction
- **Capacity**: battery capacity (Ah) - primary indicator of SoH

## File Structure

```
├── model.py                          # main entry point: command-line interface
├── processing.py                     # data loading and model training functions
├── visualization.py                  # graph generation and plotting functions
├── data.zip                          # battery dataset
├── requirements.txt                  # python dependencies
└── README.md
```

## Script Description

**`model.py`** - main entry point with command-line interface for training pipeline configuration

**`processing.py`** - data loading and model training:
   - `loadData`: loads battery data from CSV files and prepares features
   - `trainIndividualFiles`: trains models per battery with train/test split
   - `trainCombinedFiles`: trains on multiple batteries, tests on held-out file

**`visualization.py`** - graph generation and results plotting:
   - `plotPredictions`: creates scatter plots of predicted vs true capacity
   - `plotTimeSeries`: generates time series plots of battery degradation

### Features
   - **Training Modes**:
     - `individual`: single-file training with train/test split per battery
     - `combined`: multi-battery training for cross-cell generalization
   - **Feature Engineering**: configurable inclusion of cycle number for different use cases
   - **Flexible Data Splits**: customizable train/test ratios (20/80, 50/50, or any custom split)
   - **Optimized For Battery Health**: supports RUL, SoC, and SoH prediction workflows
   - **Production-Ready**: command-line interface for easy integration into ML pipelines

## Installation

The dataset could not be uploaded to this repository because the compressed file exceeds s size limits; the files can be downloaded from https://doi.org/10.1184/R1/14226830

```bash
# unzip the dataset
unzip data.zip

# install dependencies
pip install -r requirements.txt
```

## Usage

### Running The Script

```bash
# individual file training (80/20 split) with cycle number
python model.py --mode individual --include-cycle-num --test-size 0.2

# individual file training (50/50 split) with cycle number
python model.py --mode individual --include-cycle-num --test-size 0.5

# individual file training (80/20 split) without cycle number
python model.py --mode individual --test-size 0.2

# individual file training (50/50 split) without cycle number
python model.py --mode individual --test-size 0.5

# multi-file training with cycle number
python model.py --mode combined --include-cycle-num --test-file VAH07_r2.csv

# multi-file training without cycle number
python model.py --mode combined --test-file VAH01_r2.csv

# process specific files only
python model.py --mode individual --files VAH07_r2.csv VAH16_r2.csv --test-size 0.2

# custom configuration with more estimators
python model.py --mode individual --num-estimators 200 --seed 123
```

### Command-Line Arguments

- `--mode`: training mode (`individual` or `combined`)
  - `individual`: train/test split on each file separately
  - `combined`: train on multiple files, test on one held-out file
- `--include-cycle-num`: include cycle number as input feature (flag, default: false)
- `--test-size`: test set proportion for individual mode (default: 0.2)
- `--test-file`: held-out test file for combined mode (default: VAH07_r2.csv)
- `--num-estimators`: number of trees in random forest (default: 100)
- `--seed`: random seed for reproducibility (default: 42)
- `--files`: specific files to process (default: all 22 battery files)

### Model Parameters

- **Algorithm**: random forest regressor (suitable for real-time inference)
- **Number of Estimators**: 100 (configurable via `--num-estimators`)
- **Random Seed**: 42 (configurable via `--seed`)
- **Feature Engineering**: temporal normalization with optional cycle number inclusion
- **Training Mode**: individual or combined file training
- **Data Splits**: flexible train/test ratios for validation

### Output

1. **Performance Metrics**: Mean Squared Error (MSE) & R² Score: Model Validation
2. **Battery Health Indicators**:
   - Remaining Useful Life (RUL) Predictions
   - State of Charge (SoC) Estimation Accuracy
   - State of Health (SoH) Degradation Tracking
3. **Visualization Plots**:
   - Predicted Vs True Capacity Scatter Plot (Confidence Intervals)
   - Degradation Curves Over Cycling History
   - Real-Time Inference Performance Metrics

## Results Interpretation

- **Feature Engineering Impact**: toggle `--include-cycle-num` to compare models with and without cycle history
- **Training Strategy Comparison**: switch between `individual` and `combined` modes to evaluate different approaches
- **Data Split Sensitivity**: adjust `--test-size` to assess model robustness across different validation strategies
- **Cross-Cell Generalization**: use `combined` mode to evaluate performance across different battery chemistries
- **Real-Time Performance**: assess inference speed and accuracy for onboard integration requirements
- **Degradation Modeling**: analyze RUL prediction accuracy and SoH estimation reliability
- **Temporal Analysis**: understanding how model performance evolves with battery aging
- **Hyperparameter Tuning**: experiment with `--num-estimators` and `--seed` for optimization

## Dependencies

- `numpy`: high-performance numerical computations for signal processing
- `pandas`: advanced data manipulation and time-series analysis
- `matplotlib`: professional visualization for battery performance analysis
- `scikit-learn`: machine learning algorithms optimized for regression tasks
- `tqdm`: progress monitoring for large-scale battery data processing

## Future Enhancements

- **Physics-Informed Neural Networks (PINNs)**: incorporating electrochemical constraints into deep learning models
- **PyTorch Integration**: advanced neural architectures for complex degradation pattern recognition
- **Real-Time Optimization**: edge computing deployment for ultra-low latency inference
- **Multi-Modal Fusion**: integration with thermal imaging and acoustic monitoring data
- **Uncertainty Quantification**: bayesian approaches for reliable confidence intervals in safety-critical applications
- **Transfer Learning**: domain adaptation across different battery chemistries and form factors
- **Federated Learning**: orivacy-preserving model updates across distributed battery fleets
