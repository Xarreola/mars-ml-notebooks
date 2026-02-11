from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class TrainingResult:
    model: LogisticRegression
    auc: float
    class_balance: float
    n_train: int


def fit_logistic_model(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> TrainingResult:
    y = y.astype(np.uint8)
    model = LogisticRegression(
        random_state=random_state,
        max_iter=250,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1,
    )
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    return TrainingResult(
        model=model,
        auc=auc,
        class_balance=float(y.mean()),
        n_train=int(y.shape[0]),
    )
