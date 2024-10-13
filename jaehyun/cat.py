import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

from data_loader import data_loader
from submission import submit
from utils import test, raw_cv

x_train, x_val, y_train, y_val = data_loader(
    "./dataset/train/train_v1.csv",
    output_size=1,
    train_percentage=1,
)
for item in y_train.keys():
    y_train[item] = np.ravel(y_train[item])

cat_params = {
    '깐마늘(국산)': {'depth': 11, 'learning_rate': 0.2151110553042793, 'iterations': 2680, 'l2_leaf_reg': 1.7526824597185874e-08, 'border_count': 44, 'bagging_temperature': 0.6397815338916675, 'random_strength': 4.954066082948334, 'grow_policy': 'Lossguide', 'min_data_in_leaf': 97, 'max_leaves': 58},
    '건고추': {'depth': 9, 'learning_rate': 0.2982380892853731, 'iterations': 271, 'l2_leaf_reg': 0.5903164170075648, 'border_count': 245, 'bagging_temperature': 0.01688593892850543, 'random_strength': 9.560235555266887, 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 116},
    '양파': {'depth': 15, 'learning_rate': 0.06141894711037887, 'iterations': 247, 'l2_leaf_reg': 1.4074554406929478e-08, 'border_count': 29, 'bagging_temperature': 0.038157817269630995, 'random_strength': 9.653938048362608, 'grow_policy': 'Lossguide', 'min_data_in_leaf': 10, 'max_leaves': 111},
    '감자': {'depth': 5, 'learning_rate': 0.23355620279889, 'iterations': 206, 'l2_leaf_reg': 2.072820795232817e-05, 'border_count': 33, 'bagging_temperature': 0.8854201124262071, 'random_strength': 0.3706926113413769, 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 741},
    '대파': {'depth': 7, 'learning_rate': 0.004021997087502321, 'iterations': 2861, 'l2_leaf_reg': 9.890753695408794, 'border_count': 50, 'bagging_temperature': 0.7952248987337306, 'random_strength': 4.7375902728654555, 'grow_policy': 'Lossguide', 'min_data_in_leaf': 540, 'max_leaves': 127},
    '상추': {'depth': 3, 'learning_rate': 0.018165628391568156, 'iterations': 2327, 'l2_leaf_reg': 9.952970330404854, 'border_count': 5, 'bagging_temperature': 0.05750575462786251, 'random_strength': 1.9933309113463908, 'grow_policy': 'Lossguide', 'min_data_in_leaf': 995, 'max_leaves': 178},
    '배추': {'depth': 12, 'learning_rate': 0.0005042075029092138, 'iterations': 1991, 'l2_leaf_reg': 2.3630877708007642e-07, 'border_count': 24, 'bagging_temperature': 0.00105956666927301, 'random_strength': 5.235525610482856, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 9},
    '사과': {'depth': 8, 'learning_rate': 0.010323481963590734, 'iterations': 921, 'l2_leaf_reg': 0.06246857598818564, 'border_count': 21, 'bagging_temperature': 0.6263036274077526, 'random_strength': 0.0020609074922681705, 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 147},
    '무': {'depth': 10, 'learning_rate': 0.0005223469739458939, 'iterations': 2903, 'l2_leaf_reg': 1.352409718656662e-07, 'border_count': 33, 'bagging_temperature': 0.04071483786461014, 'random_strength': 1.7488754021349995, 'grow_policy': 'Depthwise', 'min_data_in_leaf': 21},
    '배': {'depth': 3, 'learning_rate': 0.05047906694130267, 'iterations': 2742, 'l2_leaf_reg': 1.2163572846286836e-07, 'border_count': 74, 'bagging_temperature': 0.6869226846473089, 'random_strength': 6.894647587995953, 'grow_policy': 'Lossguide', 'min_data_in_leaf': 12, 'max_leaves': 174}
}

models = {}
for item in x_train.keys():
    models[item] = CatBoostRegressor(**cat_params[item])

raw_cv(models, x_train, y_train, 4)

'''
submit(
    f"submission/cat_4.csv",
    "./dataset/test",
    "./sample_submission.csv",
    models,
    output_size=1
)
'''
