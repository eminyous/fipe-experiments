import time

from pathlib import Path

import argparse
import gurobipy as gp
import joblib as jl
import numpy as np
import pandas as pd

from fipe import FIPE, FeatureEncoder
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split
from utils import load, train, evaluate


def get_model(ensemble, kwargs):
    match ensemble:
        case "ab":
            model_cls = AdaBoostClassifier
            options = {"algorithm": kwargs.get("algorithm", "SAMME")}
        case "rf":
            model_cls = RandomForestClassifier
            options = {
                "max_depth": kwargs.get("max_depth", 2),
            }
        case "gb":
            model_cls = GradientBoostingClassifier
            options = {
                "max_depth": kwargs.get("max_depth", 2),
                "init": kwargs.get("init", "zero"),
            }
        case _:
            raise ValueError(f"Invalid ensemble method: {ensemble}")
    return model_cls, options


def run(
    dataset_path: Path,
    output_path: Path,
    ensemble: str,
    n_estimators: int,
    seed: int,
    norm: int,
    max_oracle_calls: int,
    timelimit: int,
    n_threads: int,
    **kwargs,
):
    data, y, _ = load(dataset_path)
    encoder = FeatureEncoder(data)
    X = encoder.X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model_cls, model_options = get_model(ensemble, kwargs)
    base, w, eps = train(
        model_cls=model_cls,
        options=model_options,
        X=X_train,
        y=y_train,
        n_estimators=n_estimators,
        seed=seed,
    )

    env = kwargs.get("env", None)
    if env is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

    pruner = FIPE(
        base=base,
        weights=w,
        encoder=encoder,
        env=env,
        eps=eps,
        max_oracle_calls=max_oracle_calls,
    )
    log_base = "-".join(
        [dataset_path.stem, ensemble, f"l{norm}", f"{n_estimators}", f"{seed}"]
    )
    log_base += "[" + \
        ",".join(f"{k}({v})" for k, v in model_options.items()) + \
        "]"
    options = {
        "max_oracle_calls": max_oracle_calls,
        "timelimit": timelimit,
        "n_threads": n_threads,
        # "eps": eps,
    }
    log_base += "[" + \
        ",".join(f"{k}({v})" for k, v in options.items()) + \
        "]"
    gurobis = output_path / "gurobi"
    gurobi_log = gurobis / f"{log_base}.log"
    pruner.build()
    pruner.set_objective(norm)
    pruner.setParam("LogFile", str(gurobi_log))
    pruner.oracle.setParam("LogFile", str(gurobi_log))
    pruner.setParam("Threads", n_threads)
    pruner.oracle.setParam("Threads", n_threads)
    if timelimit is not None:
        pruner.setParam("TimeLimit", timelimit)
    pruner.add_samples(X_train)

    start = time.time()
    pruner.prune()
    end = time.time()
    elapsed = end - start

    results = {
        "dataset": dataset_path.stem,
        "ensemble": ensemble,
        "norm": norm,
        "n.fitted.estimators": n_estimators,
        "seed": seed,
        "n.active.estimators": pruner.n_activated,
        "n.oracle.calls": pruner.n_oracle_calls,
        "elapsed.time": elapsed,
        "n.initial.samples": len(X_train),
        "n.final.samples": pruner.n_samples,
    }
    results.update(evaluate(pruner, X_test, y_test, w))
    results.update({
        "options": options,
        "ensemble.options": model_options,
    })
    csvs = output_path / "csvs"
    results_path = csvs / f"{log_base}.csv"
    pd.DataFrame([results]).to_csv(results_path, index=False)

    weights = output_path / "weights"
    weights_path = weights / f"{log_base}.csv"
    pd.DataFrame(
        {
            "t": np.arange(pruner.n_estimators),
            "base": w,
            "pruned": np.array(
                [pruner.weights[e] for e in range(pruner.n_estimators)]
            ),
        }
    ).to_csv(weights_path, index=False)


def main():
    parser = argparse.ArgumentParser("Run FIPE on a dataset")

    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")

    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all datasets in the folder",
    )

    parser.add_argument("output", type=Path, help="Output folder")

    parser.add_argument(
        "--ensemble", type=str, required=True, help="Ensemble method to use"
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
        "--seeds", type=int, nargs="+", default=[34], help="Random seed to use"
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
        "--timelimit", type=int, default=None, help="Time limit in seconds"
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        dest="n_threads",
        help="Number of threads to use",
    )

    parser.add_argument(
        "--max-depth", type=int, default=2, help="Maximum depth of the trees"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="SAMME",
        help="Algorithm to use for AdaBoost",
    )

    args = parser.parse_args()
    datasets_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()
    gurobis = output_path / "gurobi"
    gurobis.mkdir(parents=True, exist_ok=True)
    csvs = output_path / "csvs"
    csvs.mkdir(parents=True, exist_ok=True)
    weights = output_path / "weights"
    weights.mkdir(parents=True, exist_ok=True)

    if args.all:
        jl.Parallel(n_jobs=-1)(
            jl.delayed(run)(
                dataset_path=dataset_path,
                output_path=output_path,
                ensemble=args.ensemble,
                n_estimators=n_estimators,
                seed=seed,
                norm=args.norm,
                max_oracle_calls=args.max_oracle_calls,
                timelimit=args.timelimit,
                n_threads=args.n_threads,
                max_depth=args.max_depth,
                algorithm=args.algorithm,
            )
            for n_estimators in args.n_estimators
            for seed in args.seeds
            for dataset_path in datasets_path.iterdir()
        )
    else:
        dataset_path = datasets_path
        jl.Parallel(n_jobs=-1)(
            jl.delayed(run)(
                dataset_path=dataset_path,
                output_path=output_path,
                ensemble=args.ensemble,
                n_estimators=n_estimators,
                seed=seed,
                norm=args.norm,
                max_oracle_calls=args.max_oracle_calls,
                timelimit=args.timelimit,
                n_threads=args.n_threads,
                max_depth=args.max_depth,
                algorithm=args.algorithm,
            )
            for n_estimators in args.n_estimators
            for seed in args.seeds
        )


if __name__ == "__main__":
    main()
