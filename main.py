import optuna
from sklearn.datasets import make_regression
from optuna_xgb import Objective

import pandas as pd

if __name__ == "__main__":
    # X, y = make_regression(n_samples=200, n_features=1, n_targets=1, noise=30, random_state=0)

    # read a csv file
    df = pd.read_csv('dataset_pulito_GPP.csv')
    y = df["GPP"]

    # remove 'nomi' columns
    df.drop(columns=['nomi', 'DATE', 'PFT', 'GPP'], inplace=True)


    # https://optuna.readthedocs.io/en/stable/reference/pruners.html
    # pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)


    # https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
    # sampler = optuna.samplers.GPSampler()
    # sampler = optuna.samplers.CmaEsSampler()
    # sampler = optuna.samplers.NSGAIIISampler()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(pruner=pruner, sampler=sampler, direction="minimize")
    study.optimize(
        Objective(
            X_train=df,
            y_train=y,
            cv_folds=len(y)
        ),
        n_trials=100,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print(f"best value: {study.best_value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


