from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np

from dummy import LastPredictor
from data_loader import data_loader
from data_loader_test import data_loader_comb, data_loader_common
from submission import submit, submit_comb, submit_common_mean
from utils import test, raw_cv, cv, mean_raw_cv


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


# Custom model combining Lasso and DecisionTree
class CombinedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, no_common: bool = False):

        self.no_common = no_common
        self.mean_percentage = 1

        if not self.no_common:
            self.mean_percentage = 0
            self.common_model = LassoTreeRegressor()

        self.mean_model = LassoTreeRegressor()

    def fit(self, X, y):

        if not self.no_common:
            self.common_model.fit(X['common'], y['common'])

        self.mean_model.fit(X['mean'], y['mean'])

        return self

    def predict(self, X):

        if not self.no_common:
            common_preds = self.common_model.predict(X['common'])
        else:
            common_preds = 0

        mean_preds = self.mean_model.predict(X['mean'])

        return (1 - self.mean_percentage)*common_preds + self.mean_percentage*mean_preds

    def predict_common(self, X):
        if not self.no_common:
            return self.common_model.predict(X)
        else:
            return 0

    def predict_mean(self, X):
        return self.mean_model.predict(X)


x_train, _, y_train, _, x_common_train, _, y_common_train, _ = data_loader_common(
    train_path="./dataset/train",
    #x_comb=x_comb,
    input_size=3,
    output_size=1,
    train_percentage=1, process_method='ewma'
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])
    y_common_train[item] = np.ravel(y_common_train[item])

X = {}
y = {}
for item in y_train.keys():

    X[item] = {
        'common': x_common_train[item],
        'mean': x_train[item],
    }
    y[item] = {
        'common': y_common_train[item],
        'mean': y_train[item],
    }

'''
평균으로만
배추: 0.16004025900780863
무: 0.1285071617736003
양파: 0.09987201426304174
감자 수미: 0.1028334371355798
대파(일반): 0.14042360670436274
건고추: 0.02608806905922916
깐마늘(국산): 0.019279202147621136
상추: 0.09778326246747693
사과: 0.032398635158215755
배: 0.03330812202574061
cv nmae: 0.08405337697426767

평년, 평균 섞어서
배추: 0.10362430893471081 o
무: 0.12850735684660416 x
양파: 0.05583288400384659 x
감자 수미: 0.060383650201508016 x
대파(일반): 0.08253263502394044 o
건고추: 0.07149146098397328 x
깐마늘(국산): 0.019210295662650555 x
상추: 0.06471383535907708 x
사과: 0.07971336824599218 x
배: 0.01754187801979883 o
cv nmae: 0.06835516732821018
'''

models = {  # default value as zero
    "배추": CombinedRegressor(),
    "무": CombinedRegressor(no_common=True),
    "양파": CombinedRegressor(),
    "감자 수미": CombinedRegresHsor(),
    "대파(일반)": CombinedRegressor(),
    "건고추": CombinedRegressor(),
    "깐마늘(국산)": CombinedRegressor(no_common=True),
    "상추": CombinedRegressor(),
    "사과": CombinedRegressor(),
    "배": CombinedRegressor(),
}


#mean_raw_cv(models, X, y)

for item in x_train.keys():
    models[item].fit(X[item], y[item])

submit_common_mean(
    f"submission/input_3_common_mean_combined_ewma.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    input_size=3,
    process_method='ewma'
)
