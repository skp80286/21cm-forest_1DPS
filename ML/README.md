# ML-Based Inference of Physical Parameters from 21 cm Forest Spectra

This repository contains Python code and Jupyter notebooks to perform inference of astrophysical parameters from mock 21 cm forest spectra, using machine learning pipelines. The project is tailored for studies of the Epoch of Reionization (EoR) where the 21 cm absorption features provide key insights into the thermal and ionization history of the intergalactic medium.

## Overview

We implement three machine learning pipelines, each extracting features from noisy 21 cm forest spectra (which include instrumental effects) and using XGBoost for regression to predict two key physical parameters:

1. **Pipeline 1:** Computes the 1D power spectrum of the noisy 21 cm forest and uses it directly for inference.
2. **Pipeline 2:** Denoises the noisy spectrum using a U-Net and then computes the 1D power spectrum for inference.
3. **Pipeline 3:** Uses the encoder of the trained U-Net to extract a latent feature vector from the spectrum, which is then used for regression.

Most of the `.py` scripts support `-h` or `--help` to display usage instructions.

---

## Directory Contents

### Main Python modules
| File | Description |
|------|-------------|
| `F21DataLoader.py`           | Loads mock 21 cm forest datasets for training/testing. |
| `F21Stats.py`                | Computes statistical measures on the spectra. |
| `PS1D.py`                    | Calculates the 1D power spectrum. |
| `Scaling.py`                 | Performs data scaling / normalization. |
| `UnetModelWithDense.py`      | Defines the U-Net architecture with a dense latent layer for feature extraction. |
| `f21_inference_ps.py`        | Pipeline 1: inference using Power spectrum of 21-cm forest spectrum with added noise. |
| `f21_inference_ps_unet.py`   | Pipeline 2: inference using power spectrum of 21-cm forest spectrum denoised with U-Net. |
| `f21_inference_unet_with_dense.py` | Pipeline 3: Latent features extraction using U-Net encoder and inference. This requires large memory for loading the U-Net model and latent feature extraction. |
| `f21_predict_*`              | Scripts for training the models (U-Net model). |
| `posterior_maps_*.py`, `posterior_plot*.py` | Scripts to generate posterior plots and statistical summaries. |
| `plot_results.py`            | Visualization utilities. |

### Modular Inference Pipeline Components
| File | Description |
|------|-------------|
| `config_manager.py`          | Handles argument parsing and configuration setup for the inference pipeline. |
| `data_loader.py`             | Loads training and test data from CSV files for the inference pipeline. |
| `regression_trainer.py`      | Handles training of XGBoost regression models. |
| `model_tester.py`            | Handles making predictions with trained models. |
| `metrics_calculator.py`      | Calculates performance metrics from test results. |
| `results_plotter.py`         | Creates plots and visualizations for model results. |
| `f21_inference_with_uncert.py` | Modular implementation of the f21 inference pipeline with uncertainty quantification. |

### Testing Framework
| File | Description |
|------|-------------|
| `tests/`                     | Directory containing comprehensive unit tests for all pipeline components. |
| `tests/test_config_manager.py` | Unit tests for configuration management. |
| `tests/test_data_loader.py`  | Unit tests for data loading functionality. |
| `tests/test_regression_trainer.py` | Unit tests for regression training. |
| `tests/test_model_tester.py` | Unit tests for model testing. |
| `tests/test_metrics_calculator.py` | Unit tests for metrics calculation. |
| `tests/test_results_plotter.py` | Unit tests for results plotting. |
| `tests/run_tests.py`         | Test runner script for executing all tests. |

### Jupyter notebooks
| File | Purpose |
|------|---------|
| `analyse_ps_stats_data.ipynb`, `analysis1.ipynb` | Exploratory analysis of data and power spectrum statistics. |
| `denoised_los_analysis.ipynb`, `denoised_ps_dump_analysis.ipynb` | Analysis of denoised spectra. |
| `train_test_data_analysis.ipynb` | Investigation of training/test splits. |
| `timeseries_analysis.ipynb`, `ps_dump_analysis.ipynb` | power spectrum dump checks. |
| `visualize_results.ipynb` | Plots of model outputs and inference results. |

### Output and results
- `saved_output/`: Contains stored test results, plots, and posterior samples used in our publication.
- `output/`, `tmp_out/`: Temporary or intermediate files.

---

## Usage

Most scripts support:
```bash
python script_name.py -h
```

### Running the Modular Inference Pipeline

```bash
# Run the main inference script
python f21_inference_with_uncert.py --telescope uGMRT --t_int 50 --pstype noisy

# Run individual components
python config_manager.py --help
python data_loader.py --help
python regression_trainer.py --help
```

### Running Tests

```bash
# Run all tests
cd tests
python run_tests.py

# Run specific test modules
python -m unittest test_config_manager
python -m unittest test_data_loader
python -m unittest test_regression_trainer
```

---

## Benefits of the Modular Structure

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Individual modules can be used independently
3. **Maintainability**: Easier to modify and debug specific functionality
4. **Testability**: Each module can be tested independently with comprehensive unit tests
5. **Readability**: Clear separation of concerns makes the code easier to understand
6. **Reliability**: Unit tests ensure code quality and catch regressions

