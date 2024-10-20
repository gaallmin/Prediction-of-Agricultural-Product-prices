import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet, LassoLars
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader, scaling_data_loader
from submission import submit
from utils import test, raw_cv, cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train",
    output_size=1,
    train_percentage=1,
    ewm=True,
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*100
    models[item] = VotingRegressor(
        estimators=[
            (f'lasso_{i}', lasso[i]) for i in range(5)
        ],
    )

    models[item].fit(x_train[item], y_train[item])

raw_cv(models, x_train, y_train)

submit(
    f"submission/Lasso_multi_100.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    ewm=True,
)
