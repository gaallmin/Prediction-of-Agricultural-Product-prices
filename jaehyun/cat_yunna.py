import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader
from submission import submit
from utils import test, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/y_v2.csv",
    output_size=1,
    train_percentage=1,
    new_features=['평년 평균가격(원)']
)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

cat_params = {
    'random_state': 2024,
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'depth': 10,
    'l2_leaf_reg': 3,
}

models = {}
for item in x_train.keys():
    models[item] = CatBoostRegressor(**cat_params)
    models[item].fit(x_train[item], y_train[item])

cv(models, x_train, y_train)

'''
submit(
    f"submission/cat_4.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
'''
