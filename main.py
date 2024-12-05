import optuna
from sklearn.datasets import make_regression
from optuna_xgb import Objective

if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=1, n_targets=1, noise=30, random_state=0)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(pruner=pruner, sampler=sampler, direction="minimize")
    study.optimize(
        Objective(X_train=X, y_train=y, cv_folds=5),
        n_trials=100,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


