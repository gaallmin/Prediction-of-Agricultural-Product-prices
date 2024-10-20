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

    data_case = [
        "배추",
        "무",
        "양파",
        "감자 수미",
        "대파(일반)",
        "건고추",
        "깐마늘(국산)",
        "상추",
        "사과",
        "배",
    ]
    pred: dict = {
        "배추": np.array([]),
        "무": np.array([]),
        "양파": np.array([]),
        "감자 수미": np.array([]),
        "대파(일반)": np.array([]),
        "건고추": np.array([]),
        "깐마늘(국산)": np.array([]),
        "상추": np.array([]),
        "사과": np.array([]),
        "배": np.array([]),
    }

    lst_file = [f for f in listdir(test_folder) if isfile(test_folder+'/'+f)]
    lst_file.sort()
    lst_file_1 = lst_file[::2]
    lst_file_2 = lst_file[1::2]

    for test_file in lst_file_1:

        test = pd.read_csv(join(test_folder, test_file))

        for item in data_case[:5]:
            condition = test['품목(품종)명'] == item
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

    for test_file in lst_file_2:

        test = pd.read_csv(join(test_folder, test_file))

        for item in data_case[5:]:
            condition = test['품목명'] == item
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
    for item in data_case:
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
