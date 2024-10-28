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
    "감자",
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

def join_weather(
    data_dict,
    path: str = "./dataset/train/meta/TRAIN_기상_2018-2022.csv",
    join: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        '배추': ['가을', 'B', '순 최저상대습도'],
        '건고추': ['-', 'E', '순 평균풍속'],
        '무': ['가을', 'B', '순 강수량'],
        '양파': ['중만생종', 'K', '순 최고기온'],
    },
):

    years = ['2018', '2019', '2020', '2021', '2022']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    data = pd.read_csv(path)

    for item in join.keys():

        soon = ['상순', '중순', '하순']

        weather_feature = join[item][2:]

        new_data = {}
        product_list = []
        yyyymmsoon_list = []
        feature_list = []

        for year, month, soon in product(years, months, soon):
            product_list.append(item)
            yyyymmsoon_list.append(f'{year}{month}{soon}')

            try:
                feature_list.append(data[
                    (data['YYYYMMSOON'] == f'{year}{month}{soon}') &
                    (data['주산지 품목명'] == PRODUCT_CASE[CASE.index(item)]) &
                    (data['주산지 품종명'] == join[item][0]) &
                    (data['지역 이름'] == join[item][1])
                ][weather_feature].iloc[0].to_numpy())
            except:
                # 결측은 평균으롤 대채
                feature_list.append(data[
                    (data['주산지 품목명'] == PRODUCT_CASE[CASE.index(item)]) &
                    (data['주산지 품종명'] == join[item][0]) &
                    (data['지역 이름'] == join[item][1])
                ][weather_feature].mean().to_numpy())

        
        feature_list = np.array(feature_list)

        new_data['품목명'] = product_list
        new_data['YYYYMMSOON'] = yyyymmsoon_list

        for idx, feature in enumerate(weather_feature):
            new_data[feature] = feature_list[:, idx]
        new_data = pd.DataFrame(new_data)

        data_dict[item] = data_dict[item].join(new_data[['YYYYMMSOON'] + weather_feature].set_index('YYYYMMSOON'), on='YYYYMMSOON')

    return data_dict


def data_loader_meta(
    train_path: str = "./dataset/train",
    input_size: int = 9,
    train_percentage: float = 0.7,
    features: list = ['평균가격(원)'],
    is_month: bool = False,
    weather_path: str = "./dataset/train/meta/TRAIN_기상_2018-2022.csv",
    weather: dict = {
        # 리스트는 '품종', '지역', '피쳐' 순
        '배추': ['가을', 'B', '순 최저상대습도'],
        '건고추': ['-', 'E', '순 평균풍속'],
        '무': ['가을', 'B', '순 강수량'],
        '양파': ['중만생종', 'K', '순 최고기온'],
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

    data_dict = join_weather(
        data_dict,
        path=weather_path,
        join=weather
    )

    input = {}
    for item in CASE:
        input[item] = features
        if item in weather.keys():
            weather_feature = weather[item][2:]
            input[item] = input[item] + weather_feature
        if is_month:
            input[item] = input[item] + ['Month']

    output = ['평균가격(원)']
    for item in CASE:
        for idx in range(len_data[item] - input_size - OUTPUT_SIZE):
            x = data_dict[item].iloc[idx: idx + input_size][input[item]]
            y = data_dict[item].iloc[idx + input_size: idx + input_size + OUTPUT_SIZE][output]

            if process_method == 'ewm':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(alpha=0.4).mean()
            elif process_method == 'ewma':
                x['평균가격(원)'] = x['평균가격(원)'].ewm(span=4, adjust=False).mean()
            elif process_method == 'sma':
                x['평균가격(원)'] = x['평균가격(원)'].rolling(window=3, min_periods=1).mean()
            elif process_method == 'log':
                pass

            x = x.to_numpy()
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

    for item in CASE:
        print(x_train[item].shape)
