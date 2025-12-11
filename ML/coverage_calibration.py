#!/usr/bin/env python3
"""
Coverage calibration for an XGBoost inference model.

Implements the coverage–probability methodology of Sellentin & Starck (2019),
section 2.2, adapted to an XGBoost model that predicts two parameters
(xHI, logfX) from 256-dimensional feature vectors.

Inputs
------
1. XGBoost model file (command-line argument).
2. Test results CSV (command-line argument) containing true and predicted
   (xHI, logfX) for test points.
3. Directory of .npy feature files (command-line argument) that define a grid
   in (xHI, logfX) space. Filenames are assumed to encode the true parameter
   values.

Output
------
- A publication-quality plot C_alpha vs. alpha, written as PNG (and optionally PDF).
- A CSV of alpha, C_alpha, and binomial error bars.

Notes
-----
- The “posterior” for each test point is approximated as a bivariate Gaussian
  in (xHI, logfX) from Monte Carlo predictions obtained via interpolated
  feature vectors around that true point.
- Credible regions are taken as elliptical iso-density contours; for a
  2D Gaussian, the contour enclosing fraction alpha of posterior volume
  corresponds to Mahalanobis distance squared

      t_alpha = -2 * log(1 - alpha)

  because chi^2 with 2 d.o.f. is an exponential with mean 2.

- Coverage C_alpha is estimated as the fraction of test points whose true
  (xHI, logfX) lie inside the alpha-credible ellipse of their approximate
  posterior, following Eq. (2.1–2.3) of Sellentin & Starck (2019).
"""

import argparse
import glob
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("coverage_calibration")


def create_output_dir(args):
    output_dir = f'output/{sys.argv[0].split(sep=os.sep)[-1].rstrip(".py")}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("created " + output_dir)
    else:
        raise ValueError(f'Output directory already exists! {output_dir}')
    return output_dir

def setup_logging(output_dir, level: str = "INFO"):
    """Configure root logger with a simple, readable format.

    Parameters
    ----------
    output_dir : str
    level : str
        Logging level name ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    file_handler = logging.FileHandler(filename=f"{output_dir}/coverage_calibration.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
                        handlers=handlers,
                        level=numeric_level,
                        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.info(f"Commandline: {' '.join(sys.argv)}")


# ---------------------------------------------------------------------------
# XGBoost model wrapper
# ---------------------------------------------------------------------------

class XGBModelWrapper:
    """Thin wrapper around an XGBoost model (Booster or sklearn-style).

    This hides the differences between xgboost.Booster and sklearn's
    XGBRegressor / XGBClassifier, exposing a unified `predict` API.

    Parameters
    ----------
    model
        Loaded XGBoost model instance.
    is_booster : bool
        If True, model is a low-level xgboost.Booster; otherwise it is assumed
        to be a scikit-learn style estimator with `.predict()`.
    """

    def __init__(self, model, is_booster: bool):
        self.model = model
        self.is_booster = is_booster

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a 2D feature array.

        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Predictions of (xHI, logfX).
        """
        if features.ndim != 2:
            raise ValueError("features must be 2D (n_samples, n_features)")

        if self.is_booster:
            dmat = xgb.DMatrix(features)
            pred = self.model.predict(dmat)
        else:
            pred = self.model.predict(features)

        pred = np.asarray(pred)

        if pred.ndim == 1:
            # Single-output model; this script assumes 2 outputs, but we
            # tolerate 1D output and log a warning.
            LOGGER.warning(
                "Model output is 1D; expected 2D (xHI, logfX). "
                "Output shape: %s", pred.shape
            )
            pred = pred.reshape(-1, 1)

        return pred


def load_xgboost_model(path: str) -> XGBModelWrapper:
    """Load an XGBoost model from file.

    The loader tries two strategies:
      1. If the extension suggests an XGBoost native model (.json, .ubj,
         .model, .bin), it is loaded as xgboost.Booster.
      2. Otherwise, the file is loaded via joblib, assuming a scikit-learn
         style XGBoost estimator.

    Parameters
    ----------
    path : str
        Path to the saved model file.

    Returns
    -------
    XGBModelWrapper
        Unified prediction interface.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    RuntimeError
        If loading fails in both modes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        loaded_model = xgb.XGBRegressor()
        LOGGER.info("Loading XGBoost model %s", path)
        loaded_model.load_model(path)
        return loaded_model

        """
        if ext in {".json", ".ubj", ".model", ".bin"}:
            LOGGER.info("Loading XGBoost Booster from %s", path)
            booster = xgb.Booster()
            booster.load_model(path)
            return XGBModelWrapper(booster, is_booster=True)

        LOGGER.info("Loading model via joblib from %s", path)
        model = joblib.load(path)
        return XGBModelWrapper(model, is_booster=False)
        """

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model from {path}") from exc


# ---------------------------------------------------------------------------
# Feature grid handling
# ---------------------------------------------------------------------------

FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"  # robust float regex


@dataclass
class FeatureGridPoint:
    """Represents one (xHI, logfX) point in the feature grid."""

    xHI: float
    logfX: float
    features: np.ndarray  # shape (n_samples, n_features)


def parse_params_from_filename(filename: str, losdatapath='/user1/21cm_forest/21cmFAST_los/F21_noisy/') -> Tuple[float, float]:
    """Extract (xHI, logfX) from a .npy filename.

    The function assumes that the filename contains patterns like
        xHI{value}_logfX{value}
    e.g.
        features_xHI0.10_logfX-2.0.npy
        sim_xHI1e-1_logfX-3.5_noise0.npy

    Parameters
    ----------
    filename : str
        Basename of the file (without directory path).

    Returns
    -------
    (float, float)
        Parsed (xHI, logfX).

    Raises
    ------
    ValueError
        If no suitable pattern is found.
    """
    pattern = rf"_fX({FLOAT_RE})_xHI({FLOAT_RE})"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(
            f"Cannot parse xHI/logfX from filename: {filename}.\n"
            "Expected something like '...xHI0.10_logfX-2.0.npy'"
        )
    curr_logfX, curr_xHI = match.groups()
    ##  
    ## Override the xHI and logfX with the accurate values from the simulation data file
    ##
    #LOGGER.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
    data = np.fromfile('%sF21_signalonly_21cmFAST_200Mpc_z6.0_fX%s_xHI%s_8kHz.dat' % (losdatapath,curr_logfX,curr_xHI),dtype=np.float32)
    #logger.info(f'###data:{data[:20]}')
    curr_xHI = data[1]
    curr_logfX = data[2]
    #LOGGER.info(f'curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
    return curr_xHI, curr_logfX


def load_feature_grid(features_dir: str) -> List[FeatureGridPoint]:
    """Load all .npy feature grids from a directory.

    Parameters
    ----------
    features_dir : str
        Directory containing .npy files, each with shape
        (n_samples, n_features) and with (xHI, logfX) encoded
        in the filename.

    Returns
    -------
    list of FeatureGridPoint
        Parsed grid points with associated feature arrays.

    Raises
    ------
    RuntimeError
        If no .npy files are found, or shapes are inconsistent.
    """
    paths = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not paths:
        raise RuntimeError(f"No .npy files found in {features_dir}")

    grid: List[FeatureGridPoint] = []
    expected_dim = None

    LOGGER.info("Loading feature grid from %d files", len(paths))

    for path in paths:
        base = os.path.basename(path)
        try:
            xHI, logfX = parse_params_from_filename(base)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", base, exc)
            continue

        arr = np.load(path)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Feature file {base} must be 2D (n_samples, n_features); "
                f"got shape {arr.shape}"
            )

        _, n_features = arr.shape
        if expected_dim is None:
            expected_dim = n_features
            LOGGER.info(
                "Detected feature dimension: %d (from %s)", n_features, base
            )
        elif n_features != expected_dim:
            raise RuntimeError(
                "Inconsistent feature dimension: expected "
                f"{expected_dim}, but {base} has {n_features}"
            )

        grid.append(FeatureGridPoint(xHI=xHI, logfX=logfX, features=arr))

    if not grid:
        raise RuntimeError(
            "No valid feature grid files were loaded; check filename pattern."
        )

    LOGGER.info("Loaded %d grid points", len(grid))
    return grid


# ---------------------------------------------------------------------------
# Interpolation of features on the (xHI, logfX) grid
# ---------------------------------------------------------------------------

def find_k_nearest_grid_points(
    xHI: float,
    logfX: float,
    grid: List[FeatureGridPoint],
    k: int,
) -> List[FeatureGridPoint]:
    """Return the k nearest grid points in parameter space.

    Parameters
    ----------
    xHI : float
        Target xHI.
    logfX : float
        Target logfX.
    grid : list of FeatureGridPoint
        Available grid points.
    k : int
        Number of nearest neighbors to return.

    Returns
    -------
    list of FeatureGridPoint
        k nearest neighbors sorted by distance (ascending).
    """
    k = min(k, len(grid))
    distances = []
    for gp in grid:
        d = math.hypot(xHI - gp.xHI, logfX - gp.logfX)
        distances.append(d)

    indices = np.argsort(distances)[:k]
    return [grid[i] for i in indices]


def interpolate_feature_samples(
    xHI: float,
    logfX: float,
    grid: List[FeatureGridPoint],
    n_samples: int,
    k_neighbors: int,
    rng: np.random.Generator,
    samples_per_neighbor: int = 10,
) -> np.ndarray:
    """Generate interpolated feature samples for given (xHI, logfX).

    For each of `n_samples` synthetic observations, the routine:
      1. Finds the `k_neighbors` nearest grid points in (xHI, logfX).
      2. Draws `samples_per_neighbor` random feature vectors from each neighbor.
      3. Averages those samples per neighbor to get a single representative
         feature vector for that neighbor for this synthetic observation.
      4. Computes inverse-distance weights and forms a weighted average
         of those neighbor feature vectors.

    If the target matches a grid point exactly (distance == 0), the
    features are drawn directly from that grid point without interpolation.

    Parameters
    ----------
    xHI : float
        Target xHI.
    logfX : float
        Target logfX.
    grid : list of FeatureGridPoint
        Feature grid.
    n_samples : int
        Number of interpolated feature vectors to generate.
    k_neighbors : int
        Number of nearest neighbors for interpolation.
    rng : np.random.Generator
        Random number generator.
    samples_per_neighbor : int, optional
        Number of random samples to draw from each neighbor *per*
        interpolated feature, by default 1. If > 1, the neighbor
        contribution is the mean of `samples_per_neighbor` feature
        vectors drawn from that neighbor.

    Returns
    -------
    np.ndarray, shape (n_samples_eff, n_features)
        Interpolated feature matrix.
    """
    neighbors = find_k_nearest_grid_points(xHI, logfX, grid, k_neighbors)
    n_features = neighbors[0].features.shape[1]

    # Precompute distances and weights
    dists = np.array(
        [math.hypot(xHI - gp.xHI, logfX - gp.logfX) for gp in neighbors],
        dtype=float,
    )

    # If we hit an exact grid point, just use that one (no interpolation).
    if np.any(dists == 0.0):
        idx = int(np.argmin(dists))
        gp = neighbors[idx]
        max_rows = gp.features.shape[0]
        n_samples_eff = min(n_samples, max_rows)
        row_indices = rng.integers(0, max_rows, size=n_samples_eff)
        return gp.features[row_indices]

    # Inverse-distance weighting when no exact match.
    eps = 1e-8
    weights = 1.0 / (dists + eps)
    weights /= np.sum(weights)

    # Effective number of samples limited by smallest neighbor grid
    max_rows = min(gp.features.shape[0] for gp in neighbors)
    n_samples_eff = min(n_samples, max_rows)

    LOGGER.debug(
        "Interpolating %d samples at (xHI=%.4f, logfX=%.4f) using %d neighbors "
        "and %d samples per neighbor",
        n_samples_eff, xHI, logfX, len(neighbors), samples_per_neighbor,
    )

    samples = np.zeros((n_samples_eff, n_features), dtype=float)

    # For each neighbor, draw samples_per_neighbor random rows for each
    # of the n_samples_eff synthetic observations, average them, then
    # combine across neighbors with the interpolation weights.
    for w, gp in zip(weights, neighbors):
        n_rows = gp.features.shape[0]
        # indices shape: (n_samples_eff, samples_per_neighbor)
        idx_matrix = rng.integers(0, n_rows, size=(n_samples_eff, samples_per_neighbor))
        # gather features: (n_samples_eff, samples_per_neighbor, n_features)
        neighbor_draws = gp.features[idx_matrix]
        # average over the samples_per_neighbor dimension -> (n_samples_eff, n_features)
        neighbor_mean = neighbor_draws.mean(axis=1)
        samples += w * neighbor_mean

    return samples



# ---------------------------------------------------------------------------
# Posterior approximation & coverage computation
# ---------------------------------------------------------------------------

def gaussian_posterior_from_predictions(
    preds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate a 2D Gaussian posterior from model predictions.

    Parameters
    ----------
    preds : np.ndarray, shape (n_samples, 2)
        Monte Carlo predictions of (xHI, logfX) for fixed true parameters.

    Returns
    -------
    mean : np.ndarray, shape (2,)
        Estimated posterior mean.
    cov : np.ndarray, shape (2, 2)
        Estimated posterior covariance matrix (unbiased).
    """
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError(
            "preds must have shape (n_samples, 2); got %s" % repr(preds.shape)
        )

    mean = np.mean(preds, axis=0)
    cov = np.cov(preds.T, ddof=1)
    return mean, cov


def mahalanobis_sq(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Compute squared Mahalanobis distance for a 2D Gaussian.

    Parameters
    ----------
    x : np.ndarray, shape (2,)
        Point in parameter space.
    mean : np.ndarray, shape (2,)
        Gaussian mean vector.
    cov : np.ndarray, shape (2, 2)
        Covariance matrix.

    Returns
    -------
    float
        Squared Mahalanobis distance (x - mean)^T cov^{-1} (x - mean).
    """
    delta = x - mean
    inv_cov = np.linalg.inv(cov)
    return float(delta.T @ inv_cov @ delta)


def alpha_grid() -> np.ndarray:
    """Construct a grid of alpha values for coverage estimation.

    Following the spirit of Sellentin & Starck (2019), we use:
      - 100 values linearly spaced from 0.0 to 0.999
      - 20 values linearly spaced from 0.9525 to 0.9975

    The union is sorted and unique.

    Returns
    -------
    np.ndarray
        Sorted array of alpha values in [0, 1).
    """
    base = np.linspace(0.0, 0.999, 100, endpoint=True)
    tails = np.linspace(0.9525, 0.9975, 20, endpoint=True)
    alpha = np.unique(np.concatenate([base, tails]))
    return alpha


def chi2_threshold_for_alpha(alpha: float) -> float:
    """Return chi-square threshold for 2D Gaussian that encloses fraction alpha.

    For k = 2 degrees of freedom, the chi-square distribution is exponential:
        F(t) = 1 - exp(-t/2)

    Solving F(t) = alpha yields
        t_alpha = -2 * log(1 - alpha)

    Parameters
    ----------
    alpha : float
        Posterior volume fraction in [0, 1).

    Returns
    -------
    float
        Squared Mahalanobis distance threshold t_alpha.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1); got %f" % alpha)
    if alpha == 0.0:
        return 0.0
    return -2.0 * math.log(1.0 - alpha)


def compute_coverage(
    true_params: np.ndarray,
    distances_sq: np.ndarray,
    alphas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coverage C_alpha and binomial errors for given posteriors.

    We assume:
      - Each row corresponds to a test point j.
      - For that test point, we have already computed the squared Mahalanobis
        distance between the true parameters and the approximate posterior
        mean, using the posterior covariance from Monte Carlo predictions.

    For each alpha, we compute the chi-square threshold t_alpha and count how
    many test points satisfy d_j^2 <= t_alpha. The fraction of such test points
    is the coverage estimator C_hat(alpha). Binomial error bars follow Eq. (2.3)
    of Sellentin & Starck (2019):

        sigma(alpha) = sqrt(alpha * (1 - alpha) / N)

    with N = number of test points.

    Parameters
    ----------
    true_params : np.ndarray, shape (n_points, 2)
        True (xHI, logfX) for each test point (unused here except for sanity).
    distances_sq : np.ndarray, shape (n_points,)
        Squared Mahalanobis distance d_j^2 for each test point.
    alphas : np.ndarray, shape (n_alpha,)
        Grid of credible levels alpha in [0, 1).

    Returns
    -------
    coverage : np.ndarray, shape (n_alpha,)
        Estimated coverage C_alpha.
    errors : np.ndarray, shape (n_alpha,)
        Binomial standard deviation of coverage estimator C_alpha.
    """
    n_points = distances_sq.shape[0]
    if n_points == 0:
        raise ValueError("No test points provided for coverage computation.")

    coverage = np.zeros_like(alphas, dtype=float)
    errors = np.zeros_like(alphas, dtype=float)

    for i, a in enumerate(alphas):
        t_alpha = chi2_threshold_for_alpha(a)
        inside = distances_sq <= t_alpha
        C_hat = np.mean(inside.astype(float))
        coverage[i] = C_hat

        # Binomial standard deviation of the coverage estimator.
        errors[i] = math.sqrt(a * (1.0 - a) / n_points)

        LOGGER.debug(
            "alpha=%.4f, t_alpha=%.4f, coverage=%.4f, error=%.4f",
            a, t_alpha, C_hat, errors[i],
        )

    return coverage, errors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coverage(
    alphas: np.ndarray,
    coverage: np.ndarray,
    errors: np.ndarray,
    output_path: str,
) -> None:
    """Plot C_alpha vs alpha with ideal C_alpha = alpha line.

    Parameters
    ----------
    alphas : np.ndarray
        Credible levels.
    coverage : np.ndarray
        Measured coverage C_alpha.
    errors : np.ndarray
        Binomial error bars on coverage.
    output_path : str
        Base path for the figure ('.png' will be appended; '.pdf' optionally).
    """
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Ideal unbiased line
    ax.plot(
        alphas,
        alphas,
        linestyle="-",
        linewidth=2.0,
        label=r"Ideal: $C_\alpha = \alpha$",
    )

    # Measured coverage with error bars
    ax.errorbar(
        alphas,
        coverage,
        yerr=errors,
        fmt="o",
        markersize=3,
        linewidth=1.0,
        capsize=2,
        label="Measured coverage",
    )

    ax.set_xlabel(r"Credible level $\alpha$", fontsize=12)
    ax.set_ylabel(r"Coverage $C_\alpha$", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    png_path = output_path if output_path.lower().endswith(".png") else output_path + ".png"
    fig.savefig(png_path, dpi=300)
    LOGGER.info("Saved coverage plot to %s", png_path)

    # Optional PDF for publication
    pdf_path = output_path.replace(".png", "") + ".pdf"
    fig.savefig(pdf_path)
    LOGGER.info("Saved coverage plot (PDF) to %s", pdf_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level workflow
# ---------------------------------------------------------------------------

def load_test_results(csv_path: str) -> pd.DataFrame:
    """Load test results CSV file.

    Expected columns (can be adapted if needed):
      - 'test_xHI'
      - 'test_logfX'
      - 'xHI_pred'    (not used in coverage estimation, but logged)
      - 'logfX_pred'  (not used in coverage estimation, but logged)

    Parameters
    ----------
    csv_path : str
        Path to CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["test_xHI", "test_logfX"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Column '{col}' is required in {csv_path}, but is missing."
            )

    LOGGER.info(
        "Loaded %d test points from %s", df.shape[0], csv_path
    )
    return df


def run_coverage_pipeline(
    model_path: str,
    features_dir: str,
    output_dir: str,
    n_mocks: int = 100,
    n_posterior_samples: int = 2000,
    k_neighbors: int = 4,
    seed: int = 1234,
    grouping: int = 10,
) -> None:
    """
    Run a coverage test following the procedure:

        (i)  Draw N = n_mocks astrophysical parameter vectors
             theta_true,j from uniform priors:
                   <xHI> ~ Uniform(0, 1)
                   log10(fX) ~ Uniform(-4, +1).
        (ii) Generate realizations of latent features from these
             'true' parameters using interpolation on the pre-computed
             feature grid.
        (iii) Perform inference on each mock dataset, resulting in a set
              of `n_posterior_samples` samples for the astrophysical
              parameters theta. Here, the inference is approximated by
              running the XGBoost model on many interpolated feature
              realizations, and treating the resulting predictions as
              posterior samples.
        (iv) For a set of credibility levels alpha in [0, 1), test for
             each mock whether its true theta_true,j resides within the
             volume V_alpha enclosed by the alpha-th credible contour
             of its posterior. The coverage probability C(alpha) is the
             fraction of mocks for which the true values lie inside
             V_alpha.

    The credible regions are approximated as elliptical iso-contours of a
    2D Gaussian fit to the model's predictive sample for each mock, and
    V_alpha is defined by the chi-square threshold for a 2D Gaussian.

    Parameters
    ----------
    model_path : str
        Path to the saved XGBoost model.
    features_dir : str
        Directory containing .npy feature grid files with xHI/logfX in names.
    output_dir : str
        Directory where plots and coverage_results.csv will be written.
    n_mocks : int, optional
        Number of mock datasets (true parameter draws) to generate.
        Default is 100.
    n_posterior_samples : int, optional
        Number of approximate posterior samples (XGBoost predictions) per
        mock dataset. Default is 2000.
    k_neighbors : int, optional
        Number of nearest neighbors in (xHI, logfX) space used for feature
        interpolation. Default is 4.
    seed : int, optional
        Random seed used for reproducibility. Default is 1234.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    LOGGER.info("Starting coverage test with N=%d mocks", n_mocks)

    # (i) Load model and feature grid once
    model = load_xgboost_model(model_path)
    feature_grid = load_feature_grid(features_dir)

    true_params_list: List[np.ndarray] = []
    d2_list: List[float] = []

    # (i) Draw N = n_mocks true parameter vectors from the priors
    for j in range(n_mocks):
        xHI_true = rng.uniform(0.1, 0.9)          # <xHI> ~ Uniform(0, 1)
        log10_fX_true = rng.uniform(-3.5, -1.0)    # log10(fX) ~ Uniform(-4, +1)

        theta_true = np.array([xHI_true, log10_fX_true], dtype=float)
        true_params_list.append(theta_true)

        LOGGER.info(
            "Mock %d/%d: theta_true = (xHI=%.4f, log10(fX)=%.4f)",
            j + 1, n_mocks, xHI_true, log10_fX_true,
        )

        # (ii) Generate realizations of latent features from these true parameters
        features_mc = interpolate_feature_samples(
            xHI=xHI_true,
            logfX=log10_fX_true,
            grid=feature_grid,
            n_samples=n_posterior_samples,
            k_neighbors=k_neighbors,
            rng=rng,
            samples_per_neighbor=grouping,
        )

        LOGGER.info(f'features_mc.shape={features_mc.shape}')

        # (iii) Perform inference: approximate posterior samples of theta
        preds_mc = model.predict(features_mc)
        LOGGER.info(f'preds_mc.shape={preds_mc.shape}')
        preds_mc = np.asarray(preds_mc)[:, :2]  # keep (xHI, logfX)

        # Fit a 2D Gaussian to the predictive samples to define iso-contours
        mean, cov = gaussian_posterior_from_predictions(preds_mc)
        LOGGER.info(f'mean.shape={mean.shape}, cov.shape={cov.shape}')
        LOGGER.info(f'mean={mean}, cov={cov}')

        # Compute squared Mahalanobis distance of theta_true from the posterior mean
        d2 = mahalanobis_sq(theta_true, mean, cov)
        LOGGER.info(f'd2={d2}')
        d2_list.append(d2)

    true_params_arr = np.vstack(true_params_list)
    d2_arr = np.asarray(d2_list)

    LOGGER.info(
        "Finished generating mocks and approximate posteriors. "
        "Computing coverage on %d mocks.", n_mocks
    )

    # (iv) Evaluate coverage C(alpha) over a grid of credibility levels alpha
    alphas = alpha_grid()
    coverage, errors = compute_coverage(true_params_arr, d2_arr, alphas)

    # Save results to CSV
    out_csv = os.path.join(output_dir, "coverage_results.csv")
    df_out = pd.DataFrame(
        {
            "alpha": alphas,
            "coverage": coverage,
            "binomial_error": errors,
        }
    )
    df_out.to_csv(out_csv, index=False)
    LOGGER.info("Saved coverage results to %s", out_csv)

    # Plot C_alpha vs alpha and ideal C_alpha = alpha line
    plot_path = os.path.join(output_dir, "coverage_alpha")
    plot_coverage(alphas, coverage, errors, plot_path)

    LOGGER.info("Coverage test completed successfully with N=%d mocks.", n_mocks)

def run_coverage_pipeline_1(
    model_path: str,
    test_csv: str,
    features_dir: str,
    output_dir: str,
    n_mc_samples: int = 1000,
    k_neighbors: int = 4,
    seed: int = 1234,
    grouping: int = 10,
) -> None:
    """Run the full coverage-calibration pipeline.

    Parameters
    ----------
    model_path : str
        Path to the saved XGBoost model.
    test_csv : str
        Path to CSV with test results (true and predicted parameters).
    features_dir : str
        Directory containing .npy feature grid files.
    output_dir : str
        Directory to store output plots and CSVs.
    n_mc_samples : int, optional
        Number of Monte Carlo samples per test point for the approximate
        posterior, by default 1000 (or limited by available features).
    k_neighbors : int, optional
        Number of nearest neighbors in (xHI, logfX) space used for feature
        interpolation, by default 4.
    seed : int, optional
        Random seed, by default 1234.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    model = load_xgboost_model(model_path)
    feature_grid = load_feature_grid(features_dir)
    df_test = load_test_results(test_csv)

    true_params_list = []
    d2_list = []

    for idx, row in df_test.iterrows():
        test_xHI = float(row["test_xHI"])
        test_logfX = float(row["test_logfX"])
        if np.abs(test_xHI-0.11) > 0.01 or np.abs(test_logfX+0.3) > 0.01 : continue

        test_xHI = float(row["pred_xHI"])
        test_logfX = float(row["pred_logfX"])
        true_params = np.array([test_xHI, test_logfX], dtype=float)

        LOGGER.info(
            "Processing test point %d/%d: xHI=%.4f, logfX=%.4f",
            idx + 1, len(df_test), test_xHI, test_logfX,
        )

        # Interpolate features and predict many times to approximate the posterior
        features_mc = interpolate_feature_samples(
            test_xHI,
            test_logfX,
            feature_grid,
            n_samples=n_mc_samples,
            k_neighbors=k_neighbors,
            rng=rng,
            grouping=grouping,
        )
        preds_mc = model.predict(features_mc)

        # For robustness, keep only first two columns as (xHI, logfX)
        preds_mc = preds_mc[:, :2]

        mean, cov = gaussian_posterior_from_predictions(preds_mc)

        # Mahalanobis distance squared of the true parameters from the posterior mean
        d2 = mahalanobis_sq(true_params, mean, cov)

        true_params_list.append(true_params)
        d2_list.append(d2)

    true_params_arr = np.array(true_params_list)
    d2_arr = np.asarray(d2_list)

    LOGGER.info("Computed Mahalanobis distances for %d test points", d2_arr.size)

    alphas = alpha_grid()
    coverage, errors = compute_coverage(true_params_arr, d2_arr, alphas)

    # Save numeric results
    out_csv = os.path.join(output_dir, "coverage_results.csv")
    df_out = pd.DataFrame(
        {
            "alpha": alphas,
            "coverage": coverage,
            "binomial_error": errors,
        }
    )
    df_out.to_csv(out_csv, index=False)
    LOGGER.info("Saved coverage results to %s", out_csv)

    # Plot coverage diagram
    plot_path = os.path.join(output_dir, "coverage_alpha")
    plot_coverage(alphas, coverage, errors, plot_path)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate coverage C_alpha vs. alpha for an XGBoost model "
            "using feature-grid simulations and the method of "
            "Sellentin & Starck (2019)."
        )
    )

    parser.add_argument(
        "--model",
        default='saved_output/inference_gmrt50h/latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250709134119/f21_inference_xgb.pth',
        help="Path to the saved XGBoost model file.",
    )
    parser.add_argument(
        "--test-csv",
        default='saved_output/inference_gmrt50h/latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250709134119/test_results.csv',
        help=(
            "Path to test_results.csv containing true and predicted "
            "(xHI, logfX). Must contain columns 'test_xHI', 'test_logfX'."
        ),
    )
    parser.add_argument(
        "--features-dir",
        default='output/f21_unet_latent_dum_train_test_uGMRT_t50.0_20250720131101/latent',
        help="Directory containing .npy feature files with xHI/logfX in names.",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=1000,
        help=(
            "Number of Monte Carlo samples per test point for posterior "
            "approximation (default: 1000)."
        ),
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=4,
        help="Number of nearest feature grid neighbors for interpolation (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    parser.add_argument(
        "--grouping",
        type=int,
        default=10,
        help="Grouping of features"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="uniform",
        help="method1 for single test point, uniform for a uniform prior, ",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""
    args = parse_args()
    output_dir = create_output_dir(args)
    setup_logging(output_dir, args.log_level)

    LOGGER.info("Starting coverage calibration")
    LOGGER.info("Model file: %s", args.model)
    LOGGER.info("Test CSV: %s", args.test_csv)
    LOGGER.info("Features dir: %s", args.features_dir)
    LOGGER.info("Output dir: %s", output_dir)

    if args.method == "method1":
        run_coverage_pipeline_1(
            model_path=args.model,
            test_csv=args.test_csv,
            features_dir=args.features_dir,
            output_dir=output_dir,
            n_mc_samples=args.n_mc_samples,
            k_neighbors=args.k_neighbors,
            seed=args.seed,
            grouping=args.grouping,
        )
    else:
        run_coverage_pipeline(
            model_path=args.model,
            #test_csv=args.test_csv,
            features_dir=args.features_dir,
            output_dir=output_dir,
            #n_mc_samples=args.n_mc_samples,
            #k_neighbors=args.k_neighbors,
            #seed=args.seed,
            grouping=args.grouping,
        )


    LOGGER.info("Coverage calibration finished successfully.")


if __name__ == "__main__":
    main()
