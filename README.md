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
python run.py </path/to/dataset1> <path/to/dataset2> ... <path/to/datasetn> </path/to/output> --ensemble <ensemble> --n-estimators <n1> <n2> ... <nk> --seeds <seed1> <seed2> ... <seedn> --norm <norm>
```

where:

- `</path/to/dataset1>`, `</path/to/dataset2>`, ..., `</path/to/datasetn>` are the paths to the datasets to use.
- `</path/to/output>` is the path to the output folder.
- `<ensemble>` is the ensemble method to use. It can be one of the following:
    - `ab` for `AdaBoostClassifier`
    - `rf` for `RandomForestClassifier`
    - `gb` for `GradientBoostingClassifier`.
    - `lgbm` for `LGBMClassifier`.
    - `xgb` for `XGBClassifier`.
- `<n1>`, `<n2>`, ..., `<nk>` are the number of estimators to use for the ensemble method.
- `<seed1> <seed2> ... <seedn>` are the seeds to use for the random number generator.
- `<norm>` is the norm to use for the `FIPE` algorithm. It can be one of the following: `0` for `L0 norm` or `1` for `L1 norm`.

The output folder will contain the experiment results in CSV format inside a subfolder named `csvs`. The results can be merged into a single CSV file by running the following command:

```bash
python merge.py </path/to/csv1> </path/to/csv2> ... </path/to/csvn> </path/to/output>
```

where `</path/to/csv1> </path/to/csv2> ... </path/to/csvn>` are the paths to the CSV files to merge and `</path/to/output>` is the path to the output folder.

## Example

```bash
python run.py datasets/* outputs/ --ensemble ab --n-estimators 50 100 --seeds 34 42 --norm 1
```

To merge the results of the experiments, use the following command:

```bash
python merge.py outputs/csvs/* outputs/results.csv
```

## Clean

In the output folder, you will have multiple files. To clean the output folder, you can use the following command:

```bash
find outputs -mindepth 1 ! -regex '^outputs/csvs$' -delete
```