import numpy as np
from typing import Tuple


def compute_drift(sim_matrix: np.ndarray) -> np.ndarray:
    """Compute a simple drift score per document as 1 - max similarity to references."""
    max_sim = sim_matrix.max(axis=1)
    drift = 1.0 - max_sim
    return drift


def flag_drift(drift_scores: np.ndarray, threshold: float = 0.4) -> Tuple[np.ndarray, float]:
    """Return boolean array where True indicates drift > threshold."""
    alerts = drift_scores > threshold
    return alerts, threshold
