from sys import path
path.append('../')

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Normalization, LSTM, Dense, Flatten, Conv1D, BatchNormalization, Activation, MaxPooling1D, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from data_loader_meta import data_loader_meta
from data_loader import data_loader

def NMAE(y_true, y_pred):
    abs_error = tf.abs(y_true - y_pred)
    normalized_abs_error = abs_error / y_true
    itemwise_mean_error = tf.reduce_mean(normalized_abs_error, axis=1)
    nmae_score = tf.reduce_mean(itemwise_mean_error)
    
    return nmae_score

parser = argparse.ArgumentParser()
parser.add_argument('--item', dest='item', action='store')
args = parser.parse_args()

'''
x_train, x_val, y_train, y_val = data_loader_meta(
    process_method='ewma',
    is_month=True
)
'''
x_train, x_val, y_train, y_val = data_loader(process_method='ewma')

input_sequence_length = 9  # 평균가격과 강수량 각 9개
output_sequence_length = 3  # 예측할 미래의 평균가격 3개
num_features = 1  # 평균가격, 강수량 두 가지 입력 피처

model = Sequential([
    Conv1D(10, 3, input_shape=(input_sequence_length, num_features)),
    Flatten(),
    Dense(64),
    Dense(32),
    Dense(output_sequence_length),
])

print(model.summary())

'''
# 모델 입력 데이터의 형태 (batch_size, time_steps, features)
input_shape = (9, 3)  # 평균가격 타임시리즈의 입력 형태
input_all = Input(shape=input_shape)

input_price = tf.keras.layers.Lambda(lambda x: x[:, :, 0:1])(input_all)  # (Batch, 9, 1)
input_precip = tf.keras.layers.Lambda(lambda x: x[:, :, 1:2])(input_all)  # (Batch, 9, 1)
input_month = tf.keras.layers.Lambda(lambda x: x[:, :, 2:3])(input_all)  # (Batch, 9, 1)

# 평균가격 입력
x1 = Conv1D(10, 3)(input_price)
x1 = Flatten()(x1)
# 강수량 입력
x2 = Conv1D(10, 3)(input_precip)
x2 = Flatten()(x2)
# Month 
x3 = Conv1D(10, 3)(input_month)
x3 = Flatten()(x3)

# 두 입력을 연결 (concatenate) 후 Dense 층에 통합
x = Concatenate()([x1, x2, x3])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
# 3 스텝 출력
output = Dense(3)(x)

# 모델 정의
model = tf.keras.Model(inputs=input_all, outputs=output)
'''

# 0.21

cb_early_stopping = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=1500,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=500,
    min_lr=1e-10,
    verbose=1,
    min_delta=1e-5
)

adam_optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
model.compile(optimizer=adam_optimizer, loss=NMAE)

hist = model.fit(
    x_train[item],
    y_train[item],
    epochs=10,
    batch_size=32,
    validation_data=(x_val[item], y_val[item]),
    callbacks=[rlr, cb_early_stopping],
)

best_score = min(hist.history['val_loss'])
print(round(best_score, 5))
