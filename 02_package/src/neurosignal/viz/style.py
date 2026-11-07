"""
Figure style constants, enforced across all package plots, and the display-only gain for decoded traces.

R² is always computed on unscaled predictions; `fit_plot_gain` / `apply_plot_gain` only rescale what is drawn.
"""

from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

AXIS_LABEL_FONTSIZE = 16   # ax.set_xlabel / ax.set_ylabel
TITLE_FONTSIZE = 18        # ax.set_title / fig.suptitle
LEGEND_FONTSIZE = 11       # ax.legend fontsize


def fit_plot_gain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_intercept: bool = True,
) -> Tuple[float, float]:
    """
    Fit a 1D linear map y ≈ a + g * y_pred for plotting.

    This is intended for visualization of decoded traces only; downstream R²
    metrics should always be computed on the *unscaled* predictions.

    Returns
    -------
    a : float
        Intercept term.
    g : float
        Gain (slope) term.
    """
    y_true = np.ravel(np.asarray(y_true, dtype=np.float64))
    y_pred = np.ravel(np.asarray(y_pred, dtype=np.float64))
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0, 1.0
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(y_pred.reshape(-1, 1), y_true)
    a = float(reg.intercept_) if fit_intercept else 0.0
    g = float(reg.coef_[0])
    return a, g


def apply_plot_gain(
    y_pred: np.ndarray,
    a: float,
    g: float,
) -> np.ndarray:
    """Apply a previously fitted (a, g) map to predictions for plotting."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return a + g * y_pred
