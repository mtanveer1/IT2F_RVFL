# IT2F-RVFL: Interval Type-2 Fuzzy Random Vector Functional Link

This repository contains two implementations of the **Interval Type-2 Fuzzy Random Vector Functional Link (IT2F-RVFL)** framework:

- **IT2F-RVFL-C:** Closed-form ridge-regression output learning for **classification**.
- **IT2F-RVFL-A:** Fixed-regularization recursive ridge output learning for **nonlinear dynamical-system prediction**.

Both variants construct an interval type-2 (IT2F) membership-based randomized feature representation and a randomized hidden layer. Their output-weight learning procedures differ. The implementations in these two folders are separate experiments.

> **Scope:** This README describes the code actually included in this ZIP.

## Repository structure

```text
IT2F-RVFL/
├── README.md
├── IT2F-RVFL-C/
│   ├── t2f.py                       # IT2F feature generation
│   ├── t2f_rvfl.py                  # Closed-form RVFL classifier
│   ├── t2f_rvfl_main.py             # Classification experiment entry point
│   ├── datasets/
│   │   ├── acute_inflammation.mat
│   │   └── acute_nephritis.mat
│   └── results.txt                  # Included results
└── IT2F-RVFL-A/
    ├── t2f.py                       # IT2F feature generation
    ├── it2f_rvfl_models.py          # IT2FRVFLC 
    ├── rvfl_adaptive.py             # Recursive ridge updates
    ├── additional_dynamical_systems.py  # Simulation generators
    ├── all_dynamical_systems.py     # Dataset loader and chronological splits
    ├── selected_hyperparameters.json
    ├── run.py                       # Experiment entry point
    └── reproduced_results.csv      # Included results
```

Python-generated `__pycache__/` directories may also be present in the supplied archive; they are not required for execution.

## Requirements

Recommended: Python **3.10 or newer** (a recent Python 3.11/3.12 environment is suitable).



Install the dependencies from a terminal:

```powershell
python -m pip install numpy scipy scikit-learn torch
```

To verify the PyTorch installation (relevant to IT2F-RVFL-C):

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If using Conda, activate your intended environment before installing packages. Prefer `python -m pip` so installation targets the interpreter used to run the scripts.

## 1. IT2F-RVFL-C — classification

### Method

Given a feature matrix **X** with `D` input attributes and `Q` fuzzy rules, `t2f.py` produces an IT2F feature matrix **F** with `Q` columns. The implementation uses uncertain-mean Gaussian membership functions, lower/upper firing-strength branches, fixed randomized projections, and arithmetic midpoint aggregation.


### Dataset format

Put MATLAB `.mat` files inside `IT2F-RVFL-C/datasets/`. The script reads the first non-metadata variable in each file and assumes it is a **2-D numeric matrix**:

```text
feature_1  feature_2  ...  feature_D  class_label
...
```

- The **last column** must contain class labels.


### Run

In PowerShell, change to the C folder **before running**, because dataset and results paths are relative to the working directory:

```powershell
cd IT2F-RVFL\IT2F-RVFL-C
python t2f_rvfl_main.py
```

If already in the extracted `IT2F-RVFL` folder, use only:

```powershell
cd IT2F-RVFL-C
python t2f_rvfl_main.py
```

The script prints the dataset filenames as they are processed and writes **`results.txt`** in the same working directory. Rerunning overwrites that file.

### Current experiment settings

The actual `t2f_rvfl_main.py` bundled here uses these **single fixed settings**:

| Parameter | Value | Meaning |
|---|---:|---|
| `C` | `0.1` | Inverse ridge coefficient (`alpha = 10`) |
| `N` | `3` | Number of hidden nodes |
| `activation` | `1` | Sigmoid |
| `Q` | `5` | Number of fuzzy rules |
| Folds | `5` | Five contiguous folds |
| Fuzzy seed | `42` | Seed supplied to `t2f_f` |

These settings are **not the full hyperparameter grid** used in earlier experiments. 


## 2. IT2F-RVFL-A — nonlinear dynamical-system regression

### Method

The A implementation uses a reusable NumPy IT2F feature map. Input scaling is fitted on training data only. Its enhanced features are:

```text
H = activation([X_scaled, F] @ W_hidden + b_hidden)
Z = [X_scaled, F, H]
```

With fixed IT2F and hidden-layer parameters, `rvfl_adaptive.py` updates output weights sequentially by recursive ridge regression:

```text
P_0 = I / alpha
W_0 = 0
k_n = P_(n-1) z_n^T / (1 + z_n P_(n-1) z_n^T)
e_n = y_n - z_n W_(n-1)
W_n = W_(n-1) + k_n e_n
P_n = P_(n-1) - k_n z_n P_(n-1)
```

The script evaluates a fixed model after training; it does **not** adapt to the test targets.

### Included systems

| Name accepted by `run.py` | Generated system | Training pairs | Test pairs |
|---|---|---:|---:|
| `FengNonlinearSystem` | The discrete input/output recurrence defined in `all_dynamical_systems.py` | 4,998 | 200 |
| `Mackey-Glass` | Delayed Mackey–Glass system | 5,000 | 200 |
| `Lorenz` | Lorenz dynamics, scalar `x` prediction target | 5,000 | 200 |
| `NARMA-10` | NARMA-10 nonlinear memory system | 5,000 | 200 |


The generators create chronological training and test partitions. No external dynamical-system dataset download is required.

### Run

From the extracted `IT2F-RVFL` folder:

```powershell
cd IT2F-RVFL-A
python run.py
```

Run a subset:

```powershell
python run.py --systems Lorenz NARMA-10
```

Choose a different output filename:

```powershell
python run.py --systems Lorenz --output lorenz_results.csv
```

The four exact system names accepted after `--systems` are:

```text
FengNonlinearSystem  Mackey-Glass  Lorenz  NARMA-10
```

Default output: **`reproduced_results.csv`** in the current working directory. Running the script again with the default name overwrites the existing CSV.

### Saved hyperparameters

`selected_hyperparameters.json` stores the configurations selected in previous experiments. The current runner **reuses** these settings; it does **not** perform a fresh search.

| System | Fuzzy rules | Hidden neurons | Activation | Alpha |
|---|---:|---:|---|---:|
| FengNonlinearSystem | 15 | 225 | Radbas | `3.1622776601683795e-9` |
| Mackey-Glass | 10 | 225 | Radbas | `1e-9` |
| Lorenz | 5 | 140 | Sigmoid | `1e-9` |
| NARMA-10 | 55 | 225 | Sigmoid | `1e-4` |

The runner uses random seed **42**. The `ValidationRMSE` field is loaded from the saved JSON and is **not recomputed**. Test predictions and test metrics are recalculated.

### Metrics and outputs

The output CSV includes:

- **TestRMSE:** root mean squared error.
- **TestMAE:** mean absolute error.
- **TestNRMSE:** RMSE divided by the standard deviation of the test targets (`numpy.std`, population convention).
- **TestMAPEPercent:** percentage mean absolute error divided by `max(abs(target), 1e-8)` per target.
- **TrainTimeSec**, **UpdateTimeMsPerTrainSample**, **TestTimeSec:** measured runtimes (hardware and environment dependent).


---

**Research use:** Cite the associated IT2F-RVFL manuscript if you use these models in your experiments. Please also document the model version, dataset partitions, software versions, hyperparameters, and random seeds in any reported reproduction.

