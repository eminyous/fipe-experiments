import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser("Merge CSVs file into a single file")

    parser.add_argument(
        "csvs",
        type=Path,
        nargs="+",
        help="CSV files to merge"
    )

    parser.add_argument(
        "-r",
        action="store_true",
        default=False,
        dest="recursive",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output file"
    )

    args = parser.parse_args()
    recursive = bool(args.recursive)
    if recursive:
        csvs = list(Path(args.csvs[0]).resolve().glob("**/*.csv"))
    else:
        csvs = list(map(Path, args.csvs))
    output = Path(args.output).resolve()

    dataframes = [pd.read_csv(csv) for csv in csvs]
    dataframe = pd.concat(dataframes).reset_index(drop=True)
    dataframe.to_csv(output, index=False)


if __name__ == "__main__":
    main()
