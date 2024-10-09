import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load and split data into training and validation set, to prepare time-series data for predicting avg prices over 3 times units
# return: in nested dictionary set

def data_loader(df, target_items, time_column='시점', year_column='year', month_column='month', soon_column='순',
                feature_columns=None, label_column='평균가격(원)', lookback=9, prediction_window=3, batch_size=32):
    '''
    lookback: # of past time uses as input
    혹시나 싶어서 year, month, soon (순) 다 seperate 하게 column 처리
    '''
    if feature_columns is None:
        feature_columns = ['품종명', '거래단위', '등급', 'n_holiday']
    feature_columns.extend([year_column, month_column, soon_column])

    sequences_x, sequences_y = [], []

    # creating sequences for each target item
    for item in target_items:
        item_data = df[df['품목명'] == item].sort_values(by=[time_column])

        # dealing with time-series
        for i in range(len(item_data) - lookback - prediction_window +1):
            input_seq = item_data[feature_columns].iloc[i:i + lookback].values
            output_seq = item_data[label_column].iloc[i + lookback:i + lookback + prediction_window].values

            sequences_x.append(input_seq)
            sequences_y.append(output_seq)
    
    x = np.array(sequences_x)
    y = np.array(sequences_y)

    # split 8:2 
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # creating item specific batches
    item_batches = {}
    for item in target_items:
        item_data = df[df['품목명'] == item].sort_values(by=[time_column])
        item_x = item_data[feature_columns].values
        item_y = item_data[label_column].values
        item_batches[item] = {item_x[:batch_size], item_y[:batch_size]}

    
    return{
        'x_train': x_train,
        'y_train' : y_train,
        'x_val': x_val,
        'y_val' : y_val,
        'crops': item_batches
    }

