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

x_train, x_val, y_train, y_val = data_loader_meta(
    train_path="./dataset/train",
    input_size=3,
    train_percentage=1, process_method='ewma'
)
for item in x_train.keys():
    x_train[item] = x_train[item][:, [0, 2, 4, 5]]

    # 두 번째 컬럼 (index 1)을 one-hot 인코딩
    column_to_one_hot = x_train[item][:, 3].astype(int)
    one_hot_encoded = np.eye(np.max(column_to_one_hot) + 1)[column_to_one_hot]

    # 원래 배열에서 해당 컬럼 제거하고 one-hot 컬럼 붙이기
    data_one_hot = np.delete(x_train[item], 3, axis=1)  # 두 번째 컬럼 제거
    x_train[item] = np.hstack((data_one_hot, one_hot_encoded))  # one-hot 추가

models = {}
for item in x_train.keys():

    models[item] = ThreeLassoRegressor()
    #models[item].fit(x_train[item], y_train[item])

cv(models, x_train, y_train)

'''
submit(
    f"submission/data_v2_comb_input_3_residual_last_lasso_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=3,
    input_size=3,
    process_method='ewma'
)
'''
