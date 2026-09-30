"""Mean / covariance helpers shared by baselines and MPC."""
import numpy as np


def shrunk_cov(R: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """Sample covariance shrunk toward its diagonal. R has shape (T, N)."""
    S = np.cov(R, rowvar=False)
    S = np.atleast_2d(S)
    target = np.diag(np.diag(S))
    return (1 - shrink) * S + shrink * target


def chol_psd(S: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Lower-triangular L with L @ L.T ~= S, robust to tiny negative eigenvalues."""
    n = S.shape[0]
    for k in range(8):
        try:
            return np.linalg.cholesky(S + jitter * (10 ** k) * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
