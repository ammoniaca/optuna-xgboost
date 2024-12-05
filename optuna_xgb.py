"""
n_estimators: determines the number of trees (estimators) in the model
booster: type of model you will train
- gbtree: tree-based models for each boosting iteration.
- dart: DART (Dropouts meet Multiple Additive Regression Trees), which helps prevent overfitting by employing a dropout approach during training.
- gblinear: employs linear models.
tree_method:  crucial for optimizing both the speed of training and the performance of the model, especially when dealing with large datasets
- auto: XGBoost selects the most appropriate method based on the dataset.
- exact: utilizes an exact greedy algorithm. Best for small to medium datasets where precision is paramount.
- approx: employs a histogram-based approximation of the greedy algorithm. Ideal for larger datasets to balance performance and speed.
- hist: Uses a faster histogram optimized algorithm, suitable for most datasets due to its effective balance of memory usage and speed.
learning_rate (alias for 'eta'): controls the step size at each boosting iteration.
alpha (alias for 'reg_alpha'): controls the L1 regularization term on weights
lambda (alias for 'reg_lambda'): controls the L2 regularization term on weights
min_split_loss(alias for 'gamma'): controls the minimum loss reduction required to make a split on a leaf node of the tree.
max_depth: controls the maximum depth of a tree in the model. 
max_leaves: controls the maximum number of leaf nodes allowed for each tree in the model, influencing the tree’s depth and complexity.
max_bin: controls the maximum number of bins used for binning continuous features.
grow_policy: determines how the trees are grown during the training process.
min_child_weight: controls the minimum sum of instance weight needed in a child node.
max_delta_step: controls the maximum delta step allowed for each tree’s weight estimation.
subsample: determines the fraction of observations to be randomly sampled for each tree during the model’s training process. 
colsample_bynode: controls the fraction of features (columns) sampled for each node of the tree.
colsample_bylevel: controls the fraction of features (columns) sampled for each level (depth) of the tree.
colsample_bytree: controls the fraction of features (columns) sampled for each tree. 

# Dart
sample_type: the type of sampling algorithm ["uniform", "weighted"].
normalize_type: The type of normalization algorithm ["tree", "forest"].
rate_drop: the fraction of trees to drop during each boosting iteration.
skip_drop: the probability of skipping the dropout procedure during a boosting iteration.


sampling_method: plays a critical role in how training data is sampled when building trees.
max_cat_to_onehot: controls how the algorithm handles categorical features (enable_categorical is set to True). 
max_cat_threshold: controls the maximum number of categories considered for each split when using categorical features.
max_cat_threshold: controls the maximum number of categories considered for each split when using categorical features.

"""
import optuna
from optuna import Trial
from pandas import DataFrame, Series
import numpy as np

import xgboost as xgb


class Objective:
    def __init__(self,
                 X_train: DataFrame,
                 y_train: np.ndarray | Series,
                 cv_folds: int,
                 evaluation_score: str = 'rmse',
                 set_shuffle: bool = False,
                 set_stratified: bool = False,
                 set_seed: int = None):
        # Hold this implementation specific arguments as the fields of the class.
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.evaluation_score = evaluation_score
        self.set_shuffle = set_shuffle
        self.set_stratified = set_stratified
        self.set_seed = set_seed

    def __call__(self, trial):
        param_grid = {
            # used for regression tasks when the target variable is continuous.
            "objective": "reg:squarederror",
            # number of trees (estimators) in the model
            # "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, step=100),
            # type of model you will train
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            # utilizes exact where precision is paramount.
            "tree_method": "exact",
            # "tree_method": trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
            # step size at each boosting iteration.
            "learning_rate": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # minimum loss reduction required to make a split on a leaf node of the tree
            "min_split_loss": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int("max_depth", 6, 12, step=1),
            # maximum number of leaf nodes allowed for each tree in the model, influencing the tree’s depth and complexity
            "max_leaves": trial.suggest_int("max_leaves", 0, 1000, step=10),
            # maximum number of bins used for binning continuous features.
            "max_bin": trial.suggest_int("max_bin", 64, 1024, step=64),
            # tree growth policy during the training process
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15, step=1),
            # maximum delta step allowed for each tree’s weight estimation
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10, step=1),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0, 1, step=0.1),
            # fraction of features (columns) sampled for each level (depth) of the tree.
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.0, 1.0, step=0.1),
            # fraction of features (columns) sampled for each tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.0, 1.0, step=0.1)
        }
        if param_grid["tree_method"] != "exact":
            # fraction of features (columns) sampled for each node of the tree
            param_grid["colsample_bynode"] = trial.suggest_float("colsample_bynode", 0.0, 1.0, step=0.1)

        # only for DART (Dropouts meet Multiple Additive Regression Trees)
        if param_grid["booster"] == "dart":
            # type of sampling algorithm
            param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            # type of normalization algorithm
            param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            # fraction of trees to drop during each boosting iteration.
            param_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            # maximum number of categories considered for each split when using categorical features.
            # param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # Learning API data is passed using the DMatrix. DMatrix is an internal data structure that is
        # used by XGBoost, which is optimized for both memory efficiency and training speed.
        data_matrix_train = xgb.DMatrix(self.X_train, label=self.y_train)

        # initialize the model
        # xgb_model = xgb.XGBRegressor(**param_grid)

        # Add a Callback for XGBoost to prune unpromising trials.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"test-{self.evaluation_score}")

        trial_scores = xgb.cv(
            params=param_grid,
            dtrain=data_matrix_train,  # Data to be trained
            num_boost_round=trial.suggest_int("n_estimators", 500, 1500, step=10),  # number of boosting rounds (maximum iterations)
            early_stopping_rounds=100,  # Activates early stopping.
            nfold=self.cv_folds,  # number of folds in CV.
            stratified=self.set_stratified,  # perform stratified sampling.
            metrics=self.evaluation_score,
            shuffle=self.set_shuffle,
            callbacks=[pruning_callback],
            seed=self.set_seed,
            verbose_eval=0
        )

        trail_mean_score = trial_scores['test-rmse-mean'].mean(skipna=True)
        return trail_mean_score

# def objective(
#         trial: Trial,
#         X_train: DataFrame,
#         y_train: np.ndarray | Series,
#         cv_folds: int,
#         evaluation_score: str = 'rmse',
#         set_shuffle: bool = False,
#         set_stratified: bool = False,
#         set_seed: int = None
# ):
#     param_grid = {
#         # used for regression tasks when the target variable is continuous.
#         "objective": "reg:squarederror",
#         # number of trees (estimators) in the model
#         "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, step=100),
#         # type of model you will train
#         "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
#         # utilizes exact where precision is paramount.
#         "tree_method": "exact",
#         # "tree_method": trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
#         # step size at each boosting iteration.
#         "learning_rate": trial.suggest_float("eta", 1e-8, 1.0, log=True),
#         # L2 regularization weight.
#         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
#         # L1 regularization weight.
#         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#         # minimum loss reduction required to make a split on a leaf node of the tree
#         "min_split_loss": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
#         # maximum depth of the tree, signifies complexity of the tree.
#         "max_depth": trial.suggest_int("max_depth", 6, 12, step=1),
#         # maximum number of leaf nodes allowed for each tree in the model, influencing the tree’s depth and complexity
#         "max_leaves": trial.suggest_int("max_leaves", 0, 1000, step=10),
#         # maximum number of bins used for binning continuous features.
#         "max_bin": trial.suggest_int("max_bin", 64, 1024, step=64),
#         # tree growth policy during the training process
#         "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
#         # minimum child weight, larger the term more conservative the tree.
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 15, step=1),
#         # maximum delta step allowed for each tree’s weight estimation
#         "max_delta_step": trial.suggest_int("max_delta_step", 0, 10, step=1),
#         # sampling ratio for training data.
#         "subsample": trial.suggest_int("subsample", 1, 10, step=1),
#         # fraction of features (columns) sampled for each node of the tree
#         "colsample_bynode": trial.suggest_float("colsample_bynode", 0.0, 1.0, step=0.1),
#         # fraction of features (columns) sampled for each level (depth) of the tree.
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.0, 1.0, step=0.1),
#         # fraction of features (columns) sampled for each tree
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.0, 1.0, step=0.1)
#     }
#     #
#     if param_grid["booster"] == "dart":
#         # type of sampling algorithm
#         param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
#         # type of normalization algorithm
#         param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
#         # fraction of trees to drop during each boosting iteration.
#         param_grid["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
#         # maximum number of categories considered for each split when using categorical features.
#         # param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
#
#     # Learning API data is passed using the DMatrix. DMatrix is an internal data structure that is
#     # used by XGBoost, which is optimized for both memory efficiency and training speed.
#     data_matrix_train = xgb.DMatrix(X_train, label=y_train)
#
#     # initialize the model
#     # xgb_model = xgb.XGBRegressor(**param_grid)
#
#     # Add a Callback for XGBoost to prune unpromising trials.
#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"validation-{evaluation_score}")
#
#     trial_scores = xgb.cv(
#         params=param_grid,
#         dtrain=data_matrix_train,  # Data to be trained
#         num_boost_round=10000,  # number of boosting rounds (maximum iterations)
#         early_stopping_rounds=10000,  # Activates early stopping.
#         nfold=cv_folds,  # number of folds in CV.
#         stratified=set_stratified,  # perform stratified sampling.
#         metrics=evaluation_score,
#         shuffle=set_shuffle,
#         callbacks=[pruning_callback],
#         seed=set_seed
#     )
#
#     trail_mean_score = trial_scores['test-rmse-mean'].mean(skipna=True)
#     return trail_mean_score
#
