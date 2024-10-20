from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_loader(
    file_path: str,
    output_size: int = 3,
    train_percentage: float = 0.7,
    output_names: list = ["평균가격(원)"],
    new_features: list = [],
    ewm: bool = False,
    log: bool = False,
):

    if output_names in new_features:
        raise ValueError("output_name이 new_features안에 들어가지 않도록 하자")

    RANDOM_STATE = 9999
    INPUT_SIZE = 9

    data = pd.read_csv(file_path)
    x_train = {  # default value as zero
        "건고추": [],
        "감자": [],
        "배": [],
        "깐마늘(국산)": [],
        "무": [],
        "상추": [],
        "배추": [],
        "양파": [],
        "대파": [],
        "사과": [],
    }
    y_train = deepcopy(x_train)
    x_val = deepcopy(x_train)
    y_val = deepcopy(x_train)

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

    len_data = {  # default value as zero
        "건고추": 0,
        "감자": 0,
        "배": 0,
        "깐마늘(국산)": 0,
        "무": 0,
        "상추": 0,
        "배추": 0,
        "양파": 0,
        "대파": 0,
        "사과": 0,
    }
    dict_price = {}

    for item in data_case.keys():
        condition = data['품목명'] == item
        condition = condition & np.logical_or.reduce([
            data['품종명'] == cond_name for cond_name in data_case[item]['품종명']
        ]) & np.logical_or.reduce([
            data['거래단위'] == cond_name for cond_name in data_case[item]['거래단위']
        ]) & np.logical_or.reduce([
            data['등급'] == cond_name for cond_name in data_case[item]['등급']
        ])

        item_data = data.loc[condition][output_names + new_features].to_numpy()
        dict_price[item] = item_data

        if log:
            item_data = np.log(item_data)

        len_data[item] = item_data.shape[0]

    for item in data_case.keys():
        for idx in range(len_data[item] - INPUT_SIZE - output_size):
            x = dict_price[item][idx: idx + INPUT_SIZE, :].flatten()
            y = dict_price[item][idx + INPUT_SIZE: idx + INPUT_SIZE + output_size, 0:len(output_names)].flatten()

            if ewm:
                x = pd.DataFrame(x)
                x = x.ewm(alpha=0.4).mean().to_numpy().flatten()

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

    x_train, x_val, y_train, y_val = data_loader(data_path, output_size=1)

    print(x_train['건고추'])
    print(y_train['건고추'])
