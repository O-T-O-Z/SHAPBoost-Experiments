# SHAPBoost Experiments
All experiments that were ran for the paper "Shapley additive explanations based feature selection for regression and survival analysis through boosting".
See the [paper]() for details.

# Pre-requisites

First install all dependencies:
```bash
pip install -r requirements.txt
```

# Running feature selection experiments

To run the feature selection experiments:
```bash
python run_selection.py            -d <dataset> -t reg|surv [-m METHOD ...]
```
for sequential or
```bash
python run_selection_concurrent.py -d <dataset> -t reg|surv [-m METHOD ...] [-w WORKERS]
```
for concurrent execution.

# Running evaluation experiments

To run the evaluation experiments:
```bash
python run_evaluation.py -d <dataset> -t reg|surv   # + per-fold curves (--no-curves to skip)
python plot_curves.py    -d <dataset>               # feature-budget curves, all folds
```

# Figures and tables
```bash
python evaluation_experiment1.py   # Experiment 1: SHAPBoost with linear vs tree evaluator
python visualize_results.py        # main comparison (Figure 2 and 3) + full tables
```

Curves are no longer trimmed to the modal subset size. Each fold's curve holds the
test performance on its first 1..k selected features; a fold that stopped earlier
carries its last value forward (the subset that method would use under a budget of
k features), and k = 0 is the no-feature baseline. Every fold therefore counts at
every k. The lower panel shows the distribution of selected subset sizes.
