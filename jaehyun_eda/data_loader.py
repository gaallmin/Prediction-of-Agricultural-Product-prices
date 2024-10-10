import pandas as pd
import numpy as np


def data_loader_v1(
    file_path: str,
    output_size: int = 3
):

    INPUT_SIZE = 9

    data = pd.read_csv(file_path)

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

        dict_price[item] = data.loc[condition]['평균가격(원)'].to_numpy()

    for idx in range():
        pass


if __name__ == "__main__":

    data_path = "./dataset/train/train.csv"
    meta_local_path = "./dataset/train/meta/TRAIN_산지공판장_2018-2021.csv"
    meta_whole_path = "./dataset/train/meta/TRAIN_전국도매_2018-2021.csv"

    data_loader_v1(data_path)
