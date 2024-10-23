from os import path
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from scaler import TimeseriesMinMaxScaler, LogScaler


def data_loader(
    train_path: str = "./dataset/train",
    input_size: int = 9,
    output_size: int = 3,
    train_percentage: float = 0.7,
    output_names: list = ["평균가격(원)"],
    new_features: list = [],
    return_scaler: bool = False,
    process_method: str = 'ewm'  # 'ewm', sma', 'ewma', 'log'
):

    if output_names in new_features:
        raise ValueError("output_name이 new_features안에 들어가지 않도록 하자")

    RANDOM_STATE = 9999

    data_1 = pd.read_csv(path.join(train_path, "train_1.csv"))
    data_2 = pd.read_csv(path.join(train_path, "train_2.csv"))
    x_train = {  # default value as zero
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

    len_data = {  # default value as zero
        "건고추": 0,
        "감자 수미": 0,
        "배": 0,
        "깐마늘(국산)": 0,
        "무": 0,
        "상추": 0,
        "배추": 0,
        "양파": 0,
        "대파(일반)": 0,
        "사과": 0,
    }
    dict_price = {}

    for item in data_case[:5]:
        condition = data_1['품목(품종)명'] == item

        item_data = data_1.loc[condition][output_names].to_numpy()
        dict_price[item] = item_data

        len_data[item] = item_data.shape[0]

    for item in data_case[5:]:
        condition = data_2['품목명'] == item

        item_data = data_2.loc[condition][output_names].to_numpy()
        dict_price[item] = item_data

        len_data[item] = item_data.shape[0]

    for item in data_case:
        for idx in range(len_data[item] - input_size - output_size):
            x = dict_price[item][idx: idx + input_size, :].flatten()
            y = dict_price[item][idx + input_size: idx + input_size + output_size, 0:len(output_names)].flatten()

            if process_method == 'ewm':
                x = pd.DataFrame(x)
                x = x.ewm(alpha=0.4).mean().to_numpy().flatten()
            elif process_method == 'ewma':
                x = pd.DataFrame(x)
                x = x.ewm(span=4, adjust=False).mean().to_numpy().flatten()
            elif process_method == 'sma':
                x = pd.DataFrame(x)
                x = x.rolling(window=3, min_periods=1).mean().to_numpy().flatten()
            elif process_method == 'log':
                x = np.log(x + 1)
                y = np.log(y + 1)


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
            x_val = None
            y_val = None

    return x_train, x_val, y_train, y_val


def data_loader_comb(
    x_comb: dict,
    train_path: str = "./dataset/train",
    input_size: int = 9,
    output_size: int = 3,
    train_percentage: float = 0.7,
    output_names: list = ["평균가격(원)"],
    new_features: list = [],
    return_scaler: bool = False,
    process_method: str = 'ewm'  # 'ewm', sma', 'ewma', 'log'
):

    if output_names in new_features:
        raise ValueError("output_name이 new_features안에 들어가지 않도록 하자")

    RANDOM_STATE = 9999

    data_1 = pd.read_csv(path.join(train_path, "train_1.csv"))
    data_2 = pd.read_csv(path.join(train_path, "train_2.csv"))
    x_train = {  # default value as zero
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

    len_data = {  # default value as zero
        "건고추": 0,
        "감자 수미": 0,
        "배": 0,
        "깐마늘(국산)": 0,
        "무": 0,
        "상추": 0,
        "배추": 0,
        "양파": 0,
        "대파(일반)": 0,
        "사과": 0,
    }
    dict_price = {}

    for item in data_case[:5]:
        condition = data_1['품목(품종)명'] == item

        item_data = data_1.loc[condition][output_names].to_numpy()
        dict_price[item] = item_data

        len_data[item] = item_data.shape[0]

    for item in data_case[5:]:
        condition = data_2['품목명'] == item

        item_data = data_2.loc[condition][output_names].to_numpy()
        dict_price[item] = item_data

        len_data[item] = item_data.shape[0]

    for item in data_case:
        for idx in range(len_data[item] - input_size - output_size):

            x = np.vstack(
                [dict_price[key][idx: idx + input_size, :].flatten() for key in x_comb[item]]
            ).T
            y = dict_price[item][idx + input_size: idx + input_size + output_size, 0:len(output_names)].flatten()

            if process_method == 'ewm':
                x = pd.DataFrame(x)
                x = x.ewm(alpha=0.4).mean().to_numpy().flatten()
            elif process_method == 'ewma':
                x = pd.DataFrame(x)
                x = x.ewm(span=4, adjust=False).mean().to_numpy().flatten()
            elif process_method == 'sma':
                x = pd.DataFrame(x)
                x = x.rolling(window=3, min_periods=1).mean().to_numpy().flatten()
            elif process_method == 'log':
                x = np.log(x + 1)
                y = np.log(y + 1)

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
            x_val = None
            y_val = None

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":

    data_path = "./dataset/train/train.csv"

    x_train, x_val, y_train, y_val = data_loader()

    print(x_train['건고추'])
    print(y_train['건고추'])
