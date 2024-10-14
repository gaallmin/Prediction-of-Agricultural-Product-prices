from os import listdir
from os.path import join, isfile

import pandas as pd
import numpy as np


def submit(
    submit_name: str,
    test_folder: str,
    sample_submission_file: str,
    models: list,
    output_size: int,
    ewm: bool = False
):

    data_case = {
        "건고추": {"품종명": ["화건"], "거래단위": ["30 kg"], "등급": ["상품"]},
        "감자": {"품종명": ["감자 수미"], "거래단위": ["20키로상자"], "등급": ["상"]},
        "배": {"품종명": ["신고"], "거래단위": ["10 개"], "등급": ["상품"]},
        "깐마늘(국산)": {"품종명": ["깐마늘(국산)"], "거래단위": ["20 kg"], "등급": ["상품"]},
        "무": {"품종명": ["무"], "거래단위": ["20키로상자"], "등급": ["상"]},
        "상추": {"품종명": ["청"], "거래단위": ["100 g"], "등급": ["상품"]},
        "배추": {"품종명": ["배추"], "거래단위": ["10키로망대"], "등급": ["상"]},
        "양파": {"품종명": ["양파"], "거래단위": ["1키로"], "등급": ["상"]},
        "대파": {"품종명": ["대파(일반)"], "거래단위": ["1키로단"], "등급": ["상"]},
        "사과": {"품종명": ['홍로', '후지'], "거래단위": ['10 개'], "등급": ['상품']},
    }
    pred: dict = {
        "건고추": np.array([]),
        "감자": np.array([]),
        "배": np.array([]),
        "깐마늘(국산)": np.array([]),
        "무": np.array([]),
        "상추": np.array([]),
        "배추": np.array([]),
        "양파": np.array([]),
        "대파": np.array([]),
        "사과": np.array([])
    }

    lst_file = [f for f in listdir(test_folder) if isfile(test_folder+'/'+f)]
    lst_file.sort()
    for test_file in lst_file:

        test = pd.read_csv(join(test_folder, test_file))

        for item in data_case.keys():
            condition = test['품목명'] == item
            condition = condition & np.logical_or.reduce([
                test['품종명'] == cond_name for cond_name in data_case[item]['품종명']
            ]) & np.logical_or.reduce([
                test['거래단위'] == cond_name for cond_name in data_case[item]['거래단위']
            ]) & np.logical_or.reduce([
                test['등급'] == cond_name for cond_name in data_case[item]['등급']
            ])

            x = test.loc[condition]['평균가격(원)'].to_numpy()

            y_hat = np.array([])
            for i in range(4 - output_size):

                if ewm:
                    x_ewm = pd.DataFrame(x)
                    x_ewm = x_ewm.ewm(alpha=0.4).mean().to_numpy().flatten()
                    y_hat = np.append(y_hat, models[item].predict(np.expand_dims(x_ewm, axis=0)))
                else:
                    y_hat = np.append(y_hat, models[item].predict(np.expand_dims(x, axis=0)))

                x = np.append(x[i+1:], np.array(y_hat))

            pred[item] = np.append(pred[item], [y_hat])

    submission = pd.read_csv(sample_submission_file)
    for item in data_case.keys():
        submission[item] = pred[item]

    submission.to_csv(submit_name, index=False)


def submit_v2(
    submit_name: str,
    test_folder: str,
    sample_submission_file: str,
    models: list,
    output_size: int,
    new_features: list = [],
):

    data_case = {
        "건고추": {"품종명": ["화건"], "거래단위": ["30 kg"], "등급": ["상품"]},
        "감자": {"품종명": ["감자 수미"], "거래단위": ["20키로상자"], "등급": ["상"]},
        "배": {"품종명": ["신고"], "거래단위": ["10 개"], "등급": ["상품"]},
        "깐마늘(국산)": {"품종명": ["깐마늘(국산)"], "거래단위": ["20 kg"], "등급": ["상품"]},
        "무": {"품종명": ["무"], "거래단위": ["20키로상자"], "등급": ["상"]},
        "상추": {"품종명": ["청"], "거래단위": ["100 g"], "등급": ["상품"]},
        "배추": {"품종명": ["배추"], "거래단위": ["10키로망대"], "등급": ["상"]},
        "양파": {"품종명": ["양파"], "거래단위": ["1키로"], "등급": ["상"]},
        "대파": {"품종명": ["대파(일반)"], "거래단위": ["1키로단"], "등급": ["상"]},
        "사과": {"품종명": ['홍로', '후지'], "거래단위": ['10 개'], "등급": ['상품']},
    }
    pred: dict = {
        "건고추": np.array([]),
        "감자": np.array([]),
        "배": np.array([]),
        "깐마늘(국산)": np.array([]),
        "무": np.array([]),
        "상추": np.array([]),
        "배추": np.array([]),
        "양파": np.array([]),
        "대파": np.array([]),
        "사과": np.array([])
    }

    lst_file = [f for f in listdir(test_folder) if isfile(test_folder+'/'+f)]
    lst_file.sort()
    for test_file in lst_file:

        test = pd.read_csv(join(test_folder, test_file))

        for item in data_case.keys():
            condition = test['품목명'] == item
            condition = condition & np.logical_or.reduce([
                test['품종명'] == cond_name for cond_name in data_case[item]['품종명']
            ]) & np.logical_or.reduce([
                test['거래단위'] == cond_name for cond_name in data_case[item]['거래단위']
            ]) & np.logical_or.reduce([
                test['등급'] == cond_name for cond_name in data_case[item]['등급']
            ])

            x = test.loc[condition][['평균가격(원)'] + new_features].to_numpy()

            y_hat = np.array([])
            for i in range(4 - output_size):
                y_hat = np.append(y_hat, models[item].predict(np.expand_dims(x.flatten(), axis=0)))
                x = np.append(x[i+1:], np.array(y_hat))

            pred[item] = np.append(pred[item], [y_hat])

    submission = pd.read_csv(sample_submission_file)
    for item in data_case.keys():
        submission[item] = pred[item]

    submission.to_csv(submit_name, index=False)


if __name__ == "__main__":

    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import VotingRegressor
    from data_loader import data_loader_v1

    x_train, x_val, y_train, y_val = data_loader(
        "./dataset/train/merged_with_transaction_amount.csv",
        output_size=1
    )
    for item in y_train.keys():
        y_train[item] = np.ravel(y_train[item])

    depth = 1

    cat_params = {
        'random_state': 2024,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'depth': depth,
        'l2_leaf_reg': 3,
    }
    xgb_params = {
        'n_estimators': 1000,
        'random_state': 2024,
        "learning_rate": 0.05,
        'max_depth': depth,
    }
    models = {}
    for item in x_train.keys():
        cat = CatBoostRegressor(**cat_params)
        xgb = XGBRegressor(**xgb_params)
        models[item] = VotingRegressor(
            estimators=[('cat', cat), ('xgb', xgb)]
        )
        models[item].fit(x_train[item], y_train[item])

    submit(
        f"submission/voting_{depth}.csv",
        "./dataset/test",
        "./sample_submission.csv",
        models,
        output_size=1
    )
