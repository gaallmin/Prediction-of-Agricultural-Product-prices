from sys import path
path.append('../')

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, BatchNormalization, Activation, MaxPooling1D, Input, Concatenate, Lambda, Conv2D, InputLayer, ReLU
from tensorflow.keras.optimizers import Adam

from data_loader_meta import data_loader_meta
from data_loader import data_loader

def NMAE(y_true, y_pred):
    abs_error = tf.abs(y_true - y_pred)
    normalized_abs_error = abs_error / y_true
    itemwise_mean_error = tf.reduce_mean(normalized_abs_error, axis=1)
    nmae_score = tf.reduce_mean(itemwise_mean_error)
    
    return nmae_score

def create_model(input_size, num_timeseries):
    model = Sequential([
        InputLayer(input_shape=(input_size, num_timeseries)),            # 입력 형태를 (Batch, 9, 3, 1)로 설정
        Lambda(lambda x: tf.expand_dims(x, axis=-1)),         # 마지막 차원에 채널 차원을 추가하여 (Batch, 9, 3, 1)로 변환
        Conv2D(10, (2, num_timeseries), activation='relu', padding='same'),

        # 깊은 Conv2D 블록
        Conv2D(32, (2, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        
        Conv2D(64, (2, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        
        Conv2D(128, (2, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        
        Conv2D(256, (2, 3), padding='same'),
        BatchNormalization(),
        ReLU(),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3)                                      # 3 스텝 타임시리즈 예측
    ])

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--item', dest='item', action='store')
    args = parser.parse_args()

    INPUT_SIZE = 9

    # 0.11022
    # 0.12203
    x_train, x_val, y_train, y_val = data_loader_meta(
        train_percentage=.9,
        train_path="../dataset/train",
        process_method='ewma',
        is_month=True,
        weather_path="../dataset/train/meta/TRAIN_기상_2018-2022.csv",
        weather = {
            # 리스트는 '품종', '지역', '피쳐' 순
            '배추': ['가을', 'B', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
            '건고추': ['-', 'E', '순 강수량', '순 평균기온'],
            '무': ['가을', 'B', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
            '양파': ['중만생종', 'K', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
            '감자 수미': ['고랭지', 'C', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
            # "깐마늘(국산)': ['한지형', 'F', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
            '배': ['-', 'O', '순 강수량', '순 최고기온', '순 평균기온', '순 최저기온'],
        },
        input_size=INPUT_SIZE,
    )

    case_shape = {
        "배추": (INPUT_SIZE, 6),
        "무": (INPUT_SIZE, 7),
        "양파": (INPUT_SIZE, 6),
        "감자 수미": (INPUT_SIZE, 6),
        "대파(일반)": (INPUT_SIZE, 2),
        "건고추": (INPUT_SIZE, 4),
        "깐마늘(국산)": (INPUT_SIZE, 2),
        "상추": (INPUT_SIZE, 2),
        "사과": (INPUT_SIZE, 2),
        "배": (INPUT_SIZE, 4),
    }

    model = create_model(input_size=INPUT_SIZE, num_timeseries=case_shape[args.item][1])

    SAVE_FOLDER = './saved_model/'
    cb_checkpoint = ModelCheckpoint(
        filepath=SAVE_FOLDER+args.item+"_v2.keras",
        monitor='val_loss',
        mode='min',
        verbose=0,
        save_best_only=True,
    )

    cb_early_stopping = EarlyStopping(
        monitor='loss',
        mode='min',
        patience=100,
    )

    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50,
        min_lr=1e-10,
        verbose=0,
        min_delta=1e-5
    )

    adam_optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    model.compile(optimizer=adam_optimizer, loss=NMAE)

    item = args.item
    hist = model.fit(
        x_train[item],
        y_train[item],
        epochs=1500,
        batch_size=32,
        validation_data=(x_val[item], y_val[item]),
        callbacks=[rlr, cb_early_stopping, cb_checkpoint],
        verbose=1,
    )

    best_score = min(hist.history['val_loss'])
    print(round(best_score, 5))
