import optuna
from sklearn.datasets import make_regression
from optuna_xgb import Objective

import pandas as pd
from xgb_eval import xgb_regression_loocv_eval
from datetime import datetime

if __name__ == "__main__":
    # X, y = make_regression(n_samples=200, n_features=1, n_targets=1, noise=30, random_state=0)

    tag = "GPP"

    # read a csv file
    df = pd.read_csv('dataset_pulito_GPP.csv')
    y = df[tag]

    # remove 'nomi' columns
    df.drop(columns=['nomi', 'DATE', 'PFT', tag], inplace=True)

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
        n_trials=2,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    results = xgb_regression_loocv_eval(trial_best_params=trial.params, X=df.to_numpy(), y=y.to_numpy())

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%dT%H%M%S")

    metrics_value = {
        "rmse": results.get("rmse"),
        "mae": results.get("mae"),
        "max_error_score": results.get("max_error_score"),
        "r2": results.get("r2"),
        "nse": results.get("nse")
    }
    metrics_value_df = pd.DataFrame.from_dict(metrics_value, orient='index').T
    metrics_value_df.to_csv(f"best_results_{tag}_{dt_string}.csv")

    #
    best_parameters_df = pd.DataFrame.from_dict(trial.params, orient='index').T
    best_parameters_df.to_csv(f"best_parameters{tag}_{dt_string}.csv")

    value_dict = {
        'y_true': results.get("y_true"),
        'y_pred': results.get("y_pred")
    }
    df = pd.DataFrame(value_dict)
    df.to_csv(f"result_{tag}_values_{dt_string}.csv")

    # print("  Value: {}".format(trial.value))
    # print(f"best value: {study.best_value}")
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    #
