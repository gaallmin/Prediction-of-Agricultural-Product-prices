from os import path
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PRODUCT_CASE = [
    "배추",
    "무",
    "양파",
    "감자 수미",
    "대파",
    "건고추",
    "깐마늘",
    "상추",
    "사과",
    "배",
]

CASE = [
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


def preprocess_month(
    data: pd.DataFrame,
):

    data['Month'] = data['YYYYMMSOON'].apply(lambda s: int(s[4:6]))

    return data

def preprocess_weather(
    path: str = "./dataset/train/meta/TRAIN_기상_2018-2022.csv",
    features: list = ['순 평균상대습도', '순 평균기온']
):

    years = ['2018', '2019', '2020', '2021', '2022']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    soon = ['상순', '중순', '하순']

    data = pd.read_csv(path)

    new_data = {}

    features_list = {}

    for item_idx, item in enumerate(CASE):

        product_list = []
        yyyymmsoon_list = []
        for feature in features:
            features_list[feature] = []

        for year, month, soon in product(years, months, soon):
            product_list.append(item)
            yyyymmsoon_list.append(f'{year}{month}{soon}')

            for feature in features:
                features_list[feature].append(data[
                    (data['YYYYMMSOON'] == f'{year}{month}{soon}') &
                    (data['주산지 품목명'] == PRODUCT_CASE[item_idx])
                ][feature].mean())

        new_data['품목명'] = product_list
        new_data['YYYYMMSOON'] = yyyymmsoon_list
        for feature in features:
            new_data[feature] = features_list[feature]

    new_data = pd.DataFrame(new_data)

    return new_data


def data_loader_meta(
    train_path: str = "./dataset/train",
    input_size: int = 9,
    train_percentage: float = 0.7,
    is_month: bool = True,
    meta_features: dict = {
        "./dataset/train/meta/TRAIN_기상_2018-2022.csv": [
            '순 평균기온',
        ]
    },
    process_method: str = 'ewm'  # 'ewm', sma', 'ewma', 'log'
):

    OUTPUT_SIZE = 3
    RANDOM_STATE = 9999
    data_1 = pd.read_csv(path.join(train_path, "train_1.csv"))
    data_2 = pd.read_csv(path.join(train_path, "train_2.csv"))

    if is_month:  # Month as one-hot
        data_1 = preprocess_month(data_1)
        data_2 = preprocess_month(data_2)

    x_train = {
        "배추": [],
        "무": [],
        "양파": [],
        "감자 수미": [],
        "대파(일반)": [],
        "건고추": [],
        "깐마늘(국산)": [],
        "상추": [],
        "사과": [],
        "배": [],
    }
    y_train = deepcopy(x_train)
    x_val = deepcopy(x_train)
    y_val = deepcopy(x_train)

    len_data = {}
    data_dict = {}
    for item in CASE[:5]:
        condition = data_1['품목(품종)명'] == item
        data_dict[item] = data_1.loc[condition]
        len_data[item] = data_dict[item].shape[0]

    for item in CASE[5:]:
        condition = data_2['품목명'] == item
        data_dict[item] = data_2.loc[condition]
        len_data[item] = data_dict[item].shape[0]

    input = ['평균가격(원)', 'Month']
    output = ['평균가격(원)']
    for item in CASE:
        for idx in range(len_data[item] - input_size - OUTPUT_SIZE):
            x = data_dict[item].iloc[idx: idx + input_size][input]
            y = data_dict[item].iloc[idx + input_size: idx + input_size + OUTPUT_SIZE][output]

            if process_method == 'ewm':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(alpha=0.4).mean()
            elif process_method == 'ewma':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(span=4, adjust=False).mean()
            elif process_method == 'sma':
                x['평균가격(원)'] = x['평균가격(원)'].rolling(window=3, min_periods=1).mean()
            elif process_method == 'log':
                pass

            x = x.to_numpy().flatten()
            y = y.to_numpy().flatten()

            x_train[item].append(x)
            y_train[item].append(y)

        x_train[item] = np.array(x_train[item])
        y_train[item] = np.array(y_train[item])

        if train_percentage < 1:
            x_train[item], x_val[item], y_train[item], y_val[item] = train_test_split(
                x_train[item],
                y_train[item],
                test_size=1 - train_percentage,
                random_state=RANDOM_STATE
            )
        else:
            x_val[item] = None
            y_val[item] = None

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = data_loader_meta(process_method='ewma')

    print(x_train['배추'].shape)
    print(y_train['배추'].shape)
