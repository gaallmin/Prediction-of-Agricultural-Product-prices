from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

from dummy import LastPredictor
from data_loader_meta import data_loader_meta
from data_loader_test import data_loader_comb, data_loader_common, data_loader_v2
from submission import submit, submit_comb
from utils import test, raw_cv, cv
from model import ThreeLassoRegressor

x_comb = {
    "배추": ["배추"],
    "무": ["무"],
    "양파": ["양파"],
    "감자 수미": ["감자 수미"],
    "대파(일반)": ["대파(일반)"],
    "건고추": ["건고추"],
    "깐마늘(국산)": ["깐마늘(국산)"],
    "상추": ["상추"],
    "사과": ["사과"],
    "배": ["배"],
}

x_train, x_val, y_train, y_val = data_loader_meta(
    train_path="./dataset/train",
    input_size=3,
    train_percentage=1, process_method='ewma'
)

models = {}
for item in x_train.keys():

    models[item] = ThreeLassoRegressor()
    #models[item].fit(x_train[item], y_train[item])

cv(models, x_train, y_train)

'''
submit_comb(
    f"submission/data_v2_comb_input_3_residual_last_lasso_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    x_comb=x_comb,
    output_size=1,
    input_size=3,
    process_method='ewma'
)
'''
