"""Return forecasters with a shared interface, retrained walk-forward (past data only).

predict(hist_df) -> np.ndarray of next-day return forecasts, one per asset.
resid_std       -> in-sample residual std per asset (fallback interval before conformal warms up).

RidgeForecaster : numpy only, fast, the default while developing.
GRUForecaster   : PyTorch. Same interface. Requires `torch`.
"""
import numpy as np


def _windows(R: np.ndarray, L: int):
    X = np.stack([R[i:i + L].ravel() for i in range(len(R) - L)])
    y = R[L:]
    return X, y


class RidgeForecaster:
    def __init__(self, lookback=20, train_window=1000, retrain_every=63, alpha=10.0):
        self.L, self.train_window, self.retrain_every, self.alpha = lookback, train_window, retrain_every, alpha
        self._n_fit = None
        self.resid_std = None

    def _fit(self, R: np.ndarray):
        X, y = _windows(R, self.L)
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12
        self.my = y.mean(0)
        Xs = (X - self.mx) / self.sx
        A = Xs.T @ Xs + self.alpha * np.eye(Xs.shape[1])
        self.coef = np.linalg.solve(A, Xs.T @ (y - self.my))
        self.resid_std = (y - (Xs @ self.coef + self.my)).std(0)

    def predict(self, hist) -> np.ndarray:
        V = hist.values
        n = len(V)
        if self._n_fit is None or n - self._n_fit >= self.retrain_every:
            self._fit(V[-self.train_window:])
            self._n_fit = n
        x = (V[-self.L:].ravel() - self.mx) / self.sx
        return x @ self.coef + self.my


class GRUForecaster:
    def __init__(self, n_assets, lookback=20, hidden=32, epochs=15, lr=1e-3,
                 train_window=1000, retrain_every=63, seed=0, batch=64, weight_decay=1e-4):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        self.N, self.L, self.hidden, self.epochs, self.lr = n_assets, lookback, hidden, epochs, lr
        self.train_window, self.retrain_every, self.seed = train_window, retrain_every, seed
        self.batch, self.weight_decay = batch, weight_decay
        self._n_fit = None
        self.resid_std = None
        self.model = None

    def _build(self):
        nn = self.nn

        class Net(nn.Module):
            def __init__(s, n, h):
                super().__init__()
                s.gru = nn.GRU(n, h, batch_first=True)
                s.head = nn.Linear(h, n)

            def forward(s, x):
                out, _ = s.gru(x)
                return s.head(out[:, -1])
        return Net(self.N, self.hidden)

    def _fit(self, R: np.ndarray, n_seed: int):
        torch = self.torch
        torch.manual_seed(self.seed + n_seed)
        self.scale = R.std(0) + 1e-12
        Rs = R / self.scale
        X = np.stack([Rs[i:i + self.L] for i in range(len(Rs) - self.L)])
        y = Rs[self.L:]
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.model = self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = self.nn.MSELoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), self.batch):
                idx = perm[b:b + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            res = (self.model(Xt) - yt).numpy() * self.scale
        self.resid_std = res.std(0)

    def predict(self, hist) -> np.ndarray:
        torch = self.torch
        V = hist.values
        n = len(V)
        if self._n_fit is None or n - self._n_fit >= self.retrain_every:
            self._fit(V[-self.train_window:], n)
            self._n_fit = n
        x = torch.tensor((V[-self.L:] / self.scale)[None], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(x).numpy()[0] * self.scale


def build_forecaster(cfg, n_assets, seed=0):
    f = cfg["forecaster"]
    if f["kind"] == "gru":
        return GRUForecaster(n_assets, f["lookback"], f["gru_hidden"], f["gru_epochs"], f["gru_lr"],
                             f["train_window"], f["retrain_every"], seed,
                             f["gru_batch"], f["gru_weight_decay"])
    return RidgeForecaster(f["lookback"], f["train_window"], f["retrain_every"], f["ridge_alpha"])
