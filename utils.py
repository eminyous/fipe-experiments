from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from fipe import Pruner


def load(dataset_path: Path):
    name = dataset_path.stem
    full_path = dataset_path / f"{name}.full.csv"
    featurelist_path = dataset_path / f"{name}.featurelist.csv"
    data = pd.read_csv(full_path)

    # Read labels
    labels = data.iloc[:, -1]
    y = labels.astype("category").cat.codes
    y = np.array(y.values)

    data = data.iloc[:, :-1]
    with open(featurelist_path, mode="r", encoding="utf-8") as f:
        features = f.read().split(",")[:-1]
        f.close()

    return data, y, features


def train(
    model_cls, options: dict[str, Any], X, y, n_estimators: int, seed: int
):
    if not model_cls in [
        AdaBoostClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
    ]:
        raise ValueError(f"Invalid model class: {model_cls}")
    base = model_cls(n_estimators=n_estimators, random_state=seed, **options)
    base.fit(X, y)
    if model_cls == AdaBoostClassifier:
        w = base.estimator_weights_
        w = (w / w.max()) * 1e5
        eps = 1e-6
    else:
        w = np.ones(n_estimators) * 1e4
        eps = 1e-6
    return base, w, eps


def evaluate(pruner: Pruner, X, y, w):
    pred = pruner.ensemble.predict(X, w)
    new_pred = pruner.predict(X)
    accuracy = (pred == y).mean()
    pruner_accuracy = (new_pred == y).mean()
    fidelity = (pred == new_pred).mean()
    return {
        "accuracy.before.pruning": accuracy,
        "accuracy.after.pruning": pruner_accuracy,
        "fidelity": fidelity,
    }
