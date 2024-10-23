from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

from dummy import LastPredictor
from data_loader import data_loader, data_loader_comb
from submission import submit, submit_comb
from utils import test, raw_cv, cv

# Custom model combining Lasso and DecisionTree
class LassoTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tree_depth=5):
        self.tree_depth = tree_depth

        self.first = LastPredictor()

        lasso = [Lasso(alpha=1, tol=1e-7, selection='random')]*10
        self.second = VotingRegressor(
            estimators=[
                (f'second_{i}', lasso[i]) for i in range(10)
            ],
        )


    def fit(self, X, y):
        # Step 1: Fit the Lasso model
        self.first.fit(X, y)
        # Step 2: Calculate residuals
        residuals = y - self.first.predict(X)
        # Step 3: Fit the tree on residuals
        self.second.fit(X, residuals)
        return self

    def predict(self, X):
        # Predict using Lasso and add the residuals predicted by the tree
        first_preds = self.first.predict(X)
        second_preds = self.second.predict(X)
        return first_preds + second_preds


x_comb = {
    "배추": ["감자 수미", "배추"],
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

x_train, x_val, y_train, y_val = data_loader_comb(
    train_path="./dataset/train",
    x_comb=x_comb,
    input_size=3,
    output_size=1,
    train_percentage=1, process_method='ewma'
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    models[item] = LassoTreeRegressor(tree_depth=5)
    models[item].fit(x_train[item], y_train[item])

#cv(models, x_train, y_train)

submit_comb(
    f"submission/input_3_residual_last_lasso_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    x_comb=x_comb,
    output_size=1,
    input_size=3,
    process_method='ewma'
)
