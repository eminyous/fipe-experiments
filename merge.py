import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser("Merge CSVs file into a single file")

    parser.add_argument(
        "csvs",
        type=Path,
        nargs='+',
        help="CSV files to merge"
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output file"
    )

    args = parser.parse_args()

    dfs = [pd.read_csv(csv) for csv in args.csvs]
    df = pd.concat(dfs)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
