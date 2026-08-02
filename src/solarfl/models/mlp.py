"""Small torch MLP for day-ahead capacity-factor regression.

Torch rather than sklearn's MLPRegressor because the federated stage (D-013)
needs explicit weight access for FedAvg, and Flower's standard client wraps a
torch state_dict. Centralized, local, and federated runs must share this one
model class or the comparison is confounded by architecture.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FittedMLP:
    """Trained model plus the train-set standardization it depends on.

    mu/sigma travel with the model — predicting through any other scaling
    silently produces garbage, so they are not separable.
    """

    def __init__(self, model: MLP, mu: np.ndarray, sigma: np.ndarray):
        self.model = model
        self.mu = mu
        self.sigma = sigma

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sigma
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(Z)).numpy()


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
    hidden: tuple[int, ...] = (64, 32),
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 300,
    patience: int = 25,
) -> FittedMLP:
    """Adam + MSE, early-stopped on val MAE, best weights restored.

    Standardization stats come from the TRAIN split only — computing them on
    val would leak the evaluation distribution into training.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # constant columns (e.g. is_daylight after dropna)

    Zt = torch.from_numpy((X_train - mu) / sigma)
    yt = torch.from_numpy(y_train)

    model = MLP(n_features=X_train.shape[1], hidden=hidden)
    fitted = FittedMLP(model, mu, sigma)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_mae = np.inf
    best_state = None
    since_best = 0

    for _ in range(max_epochs):
        model.train()
        for idx in np.array_split(rng.permutation(len(Zt)),
                                  max(1, len(Zt) // batch_size)):
            opt.zero_grad()
            loss = loss_fn(model(Zt[idx]), yt[idx])
            loss.backward()
            opt.step()

        val_mae = float(np.mean(np.abs(fitted.predict(X_val) - np.asarray(y_val))))
        if val_mae < best_mae:
            best_mae, since_best = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    return fitted
