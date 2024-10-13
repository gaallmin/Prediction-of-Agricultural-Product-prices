from sys import argv

import optuna
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from data_loader import data_loader_v1


# Dictionary to store RMSE results and best hyperparameters
model_scores = []
best_hyperparams = {}

X_train, X_test, y_train, y_test= data_loader_v1("./dataset/train/train.csv", output_size=1, train_percentage=0.8)

item = argv[1]
X_train = X_train[item]
X_test = X_test[item]
y_train = y_train[item].flatten()
y_test = y_test[item].flatten()

# Define the objective functions for each model
def objective_xgboost(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'max_bin': trial.suggest_int('max_bin', 256, 1024),  # For tree optimization
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def objective_lightgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', -1, 20),  # -1 means no limit; expanded range
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),  # Increased range for flexibility
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),  # Minimum gain to make a further partition
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),  # Useful for imbalanced classes
        'cat_smooth': trial.suggest_int('cat_smooth', 0, 100),  # Smoothing parameter for categorical features
        'max_bin': trial.suggest_int('max_bin', 255, 1024),  # For histogram-based 
        'num_leaves': trial.suggest_int('num_leaves', 20, 128)  # Adjust range as needed
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def objective_catboost(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 15),  # Expanded range for deeper trees
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),  # Log scale for better exploration
        'iterations': trial.suggest_int('iterations', 100, 3000),  # Adjusted range for the number of iterations
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),  # Regularization parameter
        'border_count': trial.suggest_int('border_count', 1, 255),  # Number of splits for numerical features
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # Controls the strength of the bagging
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),  # Controls the amount of randomness in the leaf
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Lossguide', 'Depthwise']),  # Corrected values
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),  # Minimum number of samples in leaf
        'task_type': "GPU",
    }

    # Include max_leaves only if 'Lossguide' is chosen
    grow_policy = params['grow_policy']
    if grow_policy == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 31, 255)

    model = CatBoostRegressor(**params, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

model_objectives = {
    #"XGBoost": objective_xgboost,
    #"LightGBM": objective_lightgbm,
    "CatBoost": objective_catboost,
}

# Optimize each model with Optuna and store the results
for model_name, objective in model_objectives.items():
    print(f"Optimizing {model_name}...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    # Store the best result
    model_scores.append((model_name, study.best_value))
    best_hyperparams[model_name] = study.best_params

# Print all model scores
for model_name, rmse in model_scores:
    print(f"{model_name}: RMSE = {rmse:.4f}")

# Optional: Convert model_scores to a DataFrame for better readability
scores_df = pd.DataFrame(model_scores, columns=['Model', 'RMSE']).sort_values(by='RMSE')
print(scores_df)

# Print the best hyperparameters for each model
for model_name, params in best_hyperparams.items():
    print(f"\nBest hyperparameters for {model_name}:")
    print(params)
