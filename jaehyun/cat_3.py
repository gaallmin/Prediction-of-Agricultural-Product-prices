import numpy as np
from catboost import CatBoostRegressor

from data_loader import data_loader
from submission import submit, submit_v2
from utils import test

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/v1_averageFilling.csv",
    output_size=1,
    #new_features=['평년 평균가격(원)']
)

cat_params = {
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 3,
    'loss_function': 'MultiRMSE',
    'eval_metric': 'MultiRMSE',
    'task_type': 'CPU',
    'iterations': 1000,
    'od_type': 'Iter',
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'allow_const_label': True,
}

models = {}
for item in x_train.keys():
    models[item] = CatBoostRegressor(**cat_params)
    models[item].fit(x_train[item], y_train[item])

test(models, x_val, y_val)

submit_v2(
    f"submission/cat3_4.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1,
    #new_features=['평년 평균가격(원)']
)
