from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

from data_loader import data_loader
from data_loader_test import data_loader_comb, data_loader_common
from submission import submit, submit_comb
from utils import test, raw_cv, cv
from model import LastLassoRegressor

x_train, x_val, y_train, y_val = data_loader(
    train_path="./dataset/train",
    input_size=9,
    output_size=1,
    train_percentage=1, process_method='ewma'
)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():
    models[item] = LastLassoRegressor()
    models[item].fit(x_train[item], y_train[item])

#cv(models, x_train, y_train)

submit(
    f"submission/input_9_residual_last_lasso_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    input_size=9,
    process_method='ewma'
)
