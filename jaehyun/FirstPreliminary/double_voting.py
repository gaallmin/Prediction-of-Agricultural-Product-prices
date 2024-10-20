import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin

from data_loader import data_loader_v1, data_loader
from submission import submit, submit_v2
from utils import test, double_raw_cv

a_models = ["건고추", "감자", "배", "깐마늘(국산)", "무", "상추", "배추", "양파", "대파", "사과",]
cya_keys = ["건고추", "감자", "배", '무', "상추", "배추", "양파", "대파", "사과"]

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/y_v1.csv",
    output_size=1,
    output_names=['평균가격(원)', '평년 평균가격(원)']
)

a_x_train: dict = {}
a_y_train: dict = {}
cya_x_train: dict = {}
cya_y_train: dict = {}
for item in x_train.keys():
    a_x_train[item] = x_train[item]
    a_y_train[item] = y_train[item][:, 0]
    cya_x_train[item] = x_train[item]
    cya_y_train[item] = y_train[item][:, 1]

cat_depth = 1
xgb_depth = 1

# 500: 0.09277262654108845
# 600: 0.09258870439987489
# 700: 0.09262657209138672
# 1000: 0.0930
cat_params = {
    'random_state': 2024,
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'depth': cat_depth,
    'l2_leaf_reg': 3,
}
xgb_params = {
    'n_estimators': 1000,
    'random_state': 2024,
    "learning_rate": 0.05,
    'max_depth': xgb_depth,
}
a_models = {}
cya_models = {}
for item in a_x_train.keys():
    cat = CatBoostRegressor(**cat_params)
    xgb = XGBRegressor(**xgb_params)
    a_models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb)]
    )
    cya_models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb)]
    )

double_raw_cv(
    a_models,
    cya_models,
    a_x_train,
    a_y_train,
    cya_x_train,
    cya_y_train,
    4
)


'''
for item in cya_keys:
    cat = CatBoostRegressor(**cat_params)
    xgb = XGBRegressor(**xgb_params)
    cya_models[item] = VotingRegressor(
        estimators=[('cat', cat), ('xgb', xgb)]
    )

double_models = {}
for item in a_x_train.keys():
    double_models[item] = DoubleModel(
        a_model[item],
        cya_model[item]
    )
    double_models[item].fit()

print("common year average")
print(a_x_train)
print(a_y_train)
print(cya_x_train)
print(cya_y_train)
cv(cya_models, cya_x_train, cya_y_train)

test(models, x_val, y_val)

submit_v2(
    f"submission/voting_{cat_depth}_{xgb_depth}_600.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
'''
