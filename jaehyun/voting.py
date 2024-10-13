import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader
from submission import submit, submit_v2
from utils import test, raw_cv, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/y_v3.csv",
    output_size=1,
    train_percentage=1,
)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

params = {
    "건고추": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "감자": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "배": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "깐마늘(국산)": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "무": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "상추": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "배추": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 1500,
        'xgb_n_estimators': 120,
    },
    "양파": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "대파": {
        'cat_depth': 4,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
    "사과": {
        'cat_depth': 5,
        'xgb_depth': 3,
        'cat_n_estimators': 210,
        'xgb_n_estimators': 120,
    },
}

cat_params: dict = {}
xgb_params: dict = {}
for item in params.keys():
    cat_params[item] = {
        'n_estimators': params[item]['cat_n_estimators'],
        'learning_rate': 0.05,
        'depth': params[item]['cat_depth'],
        'l2_leaf_reg': 3,
        'verbose': 0,
    }
    xgb_params[item] = {
        'n_estimators': params[item]['xgb_n_estimators'],
        "learning_rate": 0.05,
        'max_depth': params[item]['xgb_depth'],
    }

models = {}
for item in x_train.keys():
    cat = CatBoostRegressor(**cat_params[item])
    xgb = XGBRegressor(**xgb_params[item])
    linear = LinearRegression()
    models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb), ('linear', linear)],
        weights=[0, 0, 1]
    )
    models[item] = LinearRegression()

    models[item].fit(x_train[item], y_train[item])

raw_cv(models, x_train, y_train)

submit(
    f"submission/linear.csv",
    "./dataset/test_y_v3",
    "./sample_submission.csv",
    models,
    output_size=1
)
