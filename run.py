import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
import joblib as jl
import numpy as np
import pandas as pd
from fipe import FIPE, FeatureEncoder, Pruner
from fipe.typing import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import evaluate, load, train

DEFAULT_MAX_DEPTH = 3
DEFAULT_LEARNING_RATE = 0.1


def get_model(
    model_type: str,
    model_options: dict[str, int | float | str | None],
) -> tuple[type, dict[str, int | float | str | None]]:
    match model_type:
        case "lgbm":
            model_cls = LGBMClassifier
            options = {
                "max_depth": model_options.get("max_depth", DEFAULT_MAX_DEPTH),
                "learning_rate": model_options.get(
                    "learning_rate", DEFAULT_LEARNING_RATE
                ),
                "verbose": -1,
            }
        case "xgb":
            model_cls = XGBClassifier
            options = {
                "max_depth": model_options.get("max_depth", DEFAULT_MAX_DEPTH),
                "learning_rate": model_options.get(
                    "learning_rate", DEFAULT_LEARNING_RATE
                ),
            }
        case "rf":
            model_cls = RandomForestClassifier
            options = {
                "max_depth": model_options.get("max_depth", DEFAULT_MAX_DEPTH),
            }
        case "ab":
            model_cls = AdaBoostClassifier
            options = {}
        case "gb":
            model_cls = GradientBoostingClassifier
            options = {
                "max_depth": model_options.get("max_depth", DEFAULT_MAX_DEPTH),
                "learning_rate": model_options.get(
                    "learning_rate", DEFAULT_LEARNING_RATE
                ),
            }
        case _:
            msg = f"Invalid ensemble method: {model_type}"
            raise ValueError(msg)
    return model_cls, options


def run(
    dataset: Path,
    output_path: Path,
    ensemble: str,
    n_estimators: int,
    seed: int,
    norm: int,
    max_oracle_calls: int,
    timelimit: int | None,
    n_threads: int,
    *,
    finite: bool = True,
    skip: bool = True,
    env: gp.Env | None = None,
    save_weights: bool = False,
    **kwargs,
) -> None:
    data, y, _ = load(dataset)
    encoder = FeatureEncoder(data)
    X = encoder.X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )
    model_cls, options = get_model(ensemble, kwargs)

    log_base = "-".join([
        dataset.stem,
        ensemble,
        f"f{int(finite)}",
        f"l{norm}",
        f"{n_estimators}",
        f"{seed}",
    ])
    log_base += "[" + "--".join(f"{k}({v})" for k, v in options.items()) + "]"
    pruner_options = {
        "max_oracle_calls": max_oracle_calls,
        "timelimit": timelimit,
        "n_threads": n_threads,
    }
    csvs = output_path / "csvs"
    results_path = csvs / f"{log_base}.csv"

    if skip and results_path.exists():
        return

    weights_dir = output_path / "weights"
    weights_path = weights_dir / f"{log_base}.csv"

    log_base += (
        "[" + "--".join(f"{k}({v})" for k, v in pruner_options.items()) + "]"
    )
    gurobis = output_path / "gurobi"
    gurobi_log = gurobis / f"{log_base}.log"

    base, weights, eps = train(
        model_cls=model_cls,
        options=options,
        X=X_train,
        y=y_train,
        n_estimators=n_estimators,
        seed=seed,
    )

    if env is None:
        env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.setParam("LogFile", str(gurobi_log))
    env.setParam("Threads", n_threads)
    env.start()

    if finite:
        pruner = Pruner(
            base=base,
            encoder=encoder,
            weights=weights,
            norm=norm,
            env=env,
        )
    else:
        pruner = FIPE(
            base=base,
            encoder=encoder,
            weights=weights,
            norm=norm,
            env=env,
            eps=eps,
            tol=1e-4,
            max_oracle_calls=max_oracle_calls,
        )

    pruner.build()

    if timelimit is not None:
        pruner.setParam("TimeLimit", timelimit)

    pruner.add_samples(X_train)

    start = time.time()
    pruner.prune()
    end = time.time()
    elapsed = end - start

    n_oracle_calls = 0
    if isinstance(pruner, FIPE):
        n_oracle_calls = pruner.n_oracle_calls

    results = {
        "dataset": dataset.stem,
        "ensemble": ensemble,
        "finite": int(finite),
        "norm": norm,
        "n.fitted.estimators": n_estimators,
        "seed": seed,
        "n.active.estimators": pruner.n_active_estimators,
        "n.oracle.calls": n_oracle_calls,
        "elapsed.time": elapsed,
        "n.initial.samples": len(X_train),
        "n.final.samples": pruner.n_samples,
    }
    results.update(evaluate(pruner, X_test, y_test, weights))
    results.update({
        "pruner.options": pruner_options,
        "ensemble.options": options,
    })
    pd.DataFrame([results]).to_csv(results_path, index=False)

    if save_weights:
        pd.DataFrame({
            "t": np.arange(pruner.n_estimators),
            "base": weights,
            "pruned": pruner.weights,
        }).to_csv(weights_path, index=False)


@dataclass
class Args:
    datasets: list[Path]
    output: Path
    ensemble: str
    n_estimators: list[int]
    seeds: list[int]
    norm: int
    max_oracle_calls: int
    timelimit: int | None
    n_threads: int
    max_depth: int
    learning_rate: float
    run_parallel: bool
    finite: bool = True
    skip: bool = True
    save_weights: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser("Run FIPE on a dataset")

    parser.add_argument(
        "dataset",
        type=Path,
        nargs="+",
        help="Path to the dataset folder",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run in parallel",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output folder",
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        required=True,
        choices=["lgbm", "xgb", "rf", "ab", "gb"],
        help="Ensemble method to use",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        required=True,
        dest="n_estimators",
        help="Number of estimators to use",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[34, 42],
        help="Random seed to use",
    )

    parser.add_argument(
        "--norm",
        type=int,
        required=True,
        help="Norm to be used for the objective function",
    )

    parser.add_argument(
        "--max-oracle-calls",
        type=int,
        default=10000,
        dest="max_oracle_calls",
        help="Maximum number of oracle calls",
    )

    parser.add_argument(
        "--timelimit",
        type=int,
        default=None,
        help="Time limit in seconds",
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        dest="n_threads",
        help="Number of threads to use",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum depth of the trees",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate of the trees",
    )

    parser.add_argument(
        "--finite",
        action="store_true",
        default=False,
        help="Use the finite oracle",
    )

    parser.add_argument(
        "--no-skip",
        action="store_false",
        default=True,
        dest="skip",
        help="Skip if the results already exist",
    )

    parser.add_argument(
        "--save-weights",
        action="store_true",
        default=False,
        help="Save the weights",
    )

    args = parser.parse_args()
    datasets = list(map(Path, args.dataset))
    datasets = [path.resolve() for path in datasets]
    return Args(
        datasets=datasets,
        output=Path(args.output).resolve(),
        ensemble=str(args.ensemble),
        n_estimators=list(map(int, args.n_estimators)),
        seeds=list(map(int, args.seeds)),
        norm=int(args.norm),
        max_oracle_calls=int(args.max_oracle_calls),
        timelimit=args.timelimit,
        n_threads=int(args.n_threads),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        run_parallel=bool(args.parallel),
        finite=bool(args.finite),
        skip=bool(args.skip),
        save_weights=bool(args.save_weights),
    )


def create_folders(path: Path) -> None:
    gurobis = path / "gurobi"
    gurobis.mkdir(parents=True, exist_ok=True)
    csvs = path / "csvs"
    csvs.mkdir(parents=True, exist_ok=True)
    weights = path / "weights"
    weights.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    create_folders(args.output)

    grid = itertools.product(
        args.datasets,
        args.n_estimators,
        args.seeds,
    )

    if args.run_parallel:
        jl.Parallel(n_jobs=-1)(
            jl.delayed(run)(
                dataset=dataset,
                output_path=args.output,
                ensemble=args.ensemble,
                n_estimators=n,
                seed=seed,
                norm=args.norm,
                max_oracle_calls=args.max_oracle_calls,
                timelimit=args.timelimit,
                n_threads=args.n_threads,
                max_depth=args.max_depth,
                finite=args.finite,
                skip=args.skip,
                save_weights=args.save_weights,
            )
            for dataset, n, seed in grid
        )
    else:
        for dataset, n, seed in grid:
            run(
                dataset=dataset,
                output_path=args.output,
                ensemble=args.ensemble,
                n_estimators=n,
                seed=seed,
                norm=args.norm,
                max_oracle_calls=args.max_oracle_calls,
                timelimit=args.timelimit,
                n_threads=args.n_threads,
                max_depth=args.max_depth,
                finite=args.finite,
                skip=args.skip,
                save_weights=args.save_weights,
            )


if __name__ == "__main__":
    main()
