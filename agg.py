from argparse import ArgumentParser
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Args:
    results: list[Path]
    recursive: bool
    output: Path


def parse_args() -> Args:
    description = "Aggregate the results of the experiments"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="The results files to aggregate",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for results files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="The output file",
    )
    args = parser.parse_args()
    return Args(
        results=list(map(Path, args.results)),
        recursive=bool(args.recursive),
        output=Path(args.output),
    )


def get_files_recursive(paths: list[Path], ext: str) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(get_files_recursive(path.iterdir(), ext))
        elif path.suffix == ext:
            files.append(path)
    return files


def get_files_flat(paths: list[Path], ext: str) -> list[Path]:
    return [path for path in paths if path.suffix == ext]


def get_files(paths: list[Path], ext: str, *, recursive: bool) -> list[Path]:
    if recursive:
        return get_files_recursive(paths, ext)
    return get_files_flat(paths, ext)


DATASET_COL = "dataset"
ENSEMBLE_COL = "ensemble"
NORM_COL = "norm"
N_ESTIMATORS_COL = "n.fitted.estimators"
N_ACTIVE_ESTIMATORS_COL = "n.active.estimators"
N_ORACLES_CALLS_COL = "n.oracle.calls"
ELAPSED_TIME_COL = "elapsed.time"
ACCURACY_AFTER_COL = "accuracy.after.pruning"
ACCURACY_COL = "accuracy"
FIDELITY_COL = "fidelity"
ENSEMBLE_OPTIONS_COL = "ensemble.options"
MAX_DEPTH_COL = "max.depth"


def _max_depth(options: str) -> int:
    return dict(literal_eval(options)).get("max_depth", 0)


def main() -> None:
    args = parse_args()
    files = get_files(args.results, ext=".csv", recursive=args.recursive)

    data = pd.concat(map(pd.read_csv, files)).reset_index(drop=True)
    data[MAX_DEPTH_COL] = data[ENSEMBLE_OPTIONS_COL].apply(_max_depth)

    pivoted = data.pivot_table(
        index=[
            DATASET_COL,
            N_ESTIMATORS_COL,
            ENSEMBLE_COL,
            NORM_COL,
            MAX_DEPTH_COL,
        ],
        values=[
            N_ACTIVE_ESTIMATORS_COL,
            N_ORACLES_CALLS_COL,
            ELAPSED_TIME_COL,
            ACCURACY_AFTER_COL,
            FIDELITY_COL,
        ],
        aggfunc={
            N_ACTIVE_ESTIMATORS_COL: "mean",
            N_ORACLES_CALLS_COL: "mean",
            ELAPSED_TIME_COL: "mean",
            ACCURACY_AFTER_COL: "mean",
            FIDELITY_COL: "mean",
        },
    ).rename(
        columns={
            ACCURACY_AFTER_COL: ACCURACY_COL,
        }
    )
    pivoted.to_csv(args.output)


if __name__ == "__main__":
    main()
