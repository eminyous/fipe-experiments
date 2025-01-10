import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fipe import Pruner
from fipe.typing import (
    AdaBoostClassifier,
    BaseEnsemble,
    GradientBoostingClassifier,
    MClass,
    MNumber,
    RandomForestClassifier,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def load(dataset_path: Path) -> tuple[pd.DataFrame, MClass, list[str]]:
    name = dataset_path.stem
    full_path = dataset_path / f"{name}.full.csv"
    featurelist_path = dataset_path / f"{name}.featurelist.csv"
    data = pd.read_csv(full_path)

    # Read labels
    labels = data.iloc[:, -1]
    y = labels.astype("category").cat.codes
    y = np.array(y.values)

    data = data.iloc[:, :-1]
    with featurelist_path.open(encoding="utf-8") as f:
        features = f.read().split(",")[:-1]
        f.close()

    return data, y, features


def train(
    model_cls: type,
    options: dict[str, int | float | str | None],
    X,
    y,
    n_estimators: int,
    seed: int,
) -> tuple[BaseEnsemble, MNumber, float]:
    if model_cls not in {
        LGBMClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        XGBClassifier,
    }:
        msg = f"Invalid model class: {model_cls}"
        raise ValueError(msg)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        base = model_cls(
            n_estimators=n_estimators,
            random_state=seed,
            **options,
        )
        base.fit(X, y)

        if isinstance(base, LGBMClassifier):
            base = base.booster_
        elif isinstance(base, XGBClassifier):
            base = base.get_booster()

    w = np.ones(n_estimators)
    eps = 1e-6
    return base, w, eps


def evaluate(pruner: Pruner, X, y, w: MNumber) -> dict[str, float]:
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
