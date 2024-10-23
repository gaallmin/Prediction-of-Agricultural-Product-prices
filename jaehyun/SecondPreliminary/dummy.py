import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from data_loader import data_loader
from submission import submit
from utils import test, raw_cv, cv

class LastPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        # Calculate the mean of each feature and store it
        self.mean_ = np.mean(X, axis=1)
        return self

    def predict(self, X):
        # Return the stored mean as a prediction for each sample

        return X[:, -1].flatten()


x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train",
    output_size=1,
    train_percentage=1,
)

for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

models = {}
for item in x_train.keys():

    models[item] = LastPredictor()
    #models[item].fit(x_train[item], y_train[item])

raw_cv(models, x_train, y_train)

'''
submit(
    f"submission/LassoTree.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
)
'''
