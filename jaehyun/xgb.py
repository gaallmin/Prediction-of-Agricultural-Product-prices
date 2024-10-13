import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from data_loader import data_loader_v1
from submission import submit
from utils import test

x_train, x_val, y_train, y_val = data_loader_v1("./dataset/train/train.csv", output_size=1)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

xgb_depth = 5

xgb_params = {
    'n_estimators': 1000,
    'random_state': 2024,
    "learning_rate": 0.05,
    'max_depth': xgb_depth,
}
models = {}
for item in x_train.keys():
    models[item] = XGBRegressor(**xgb_params)
    models[item].fit(x_train[item], y_train[item])

test(models, x_val, y_val)

'''
submit(
    f"submission/voting_{cat_depth}_{xgb_depth}.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
'''
