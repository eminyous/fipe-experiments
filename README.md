# FIPE Paper Experiments Reproducibility

This repository contains the code that allows to reproduce the results of the paper "FIPE: Functionnaly Identical Prunning Ensemble".

## Requirements

The code is written in Python 3.10. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the experiments, make sure you have a valid Gurobi license and the required libraries installed. Then, from the current folder, execute the following command:

```bash
python run.py </path/to/dataset> </path/to/output> --ensemble <ensemble> --n-estimators <n_estimators> --seeds <seed1> <seed2> ... <seedn> --norm <norm>
```

where:

- `</path/to/dataset>` is the path to the dataset file in CSV format.
- `</path/to/output>` is the path to the output folder.
- `<ensemble>` is the ensemble method to use. It can be one of the following: `ab` for `AdaBoostClassifier` or `rf` for `RandomForestClassifier` or `gb` for `GradientBoostingClassifier`.
- `<n_estimators>` is the number of estimators to use in the ensemble.
- `<seed1> <seed2> ... <seedn>` are the seeds to use for the random number generator.
- `<norm>` is the norm to use for the `FIPE` algorithm. It can be one of the following: `0` for `L0 norm`, `1` for `L1 norm`, `2` for `L2 norm`.

The output folder will contain the experiment results in CSV format inside a subfolder named `csvs`. The results can be merged into a single CSV file by running the following command:

```bash
python merge.py </path/to/csv1> </path/to/csv2> ... </path/to/csvn> </path/to/output>
```

where `</path/to/csv1> </path/to/csv2> ... </path/to/csvn>` are the paths to the CSV files to merge and `</path/to/output>` is the path to the output folder.

## Example

```bash
python run.py --all datasets/ outputs/ --ensemble ab --n-estimators 50 100 --seeds 42 47 52 --norm 1
```

To merge the results of the experiments, use the following command:

```bash
python merge.py outputs/csvs/* outputs/results.csv
```
