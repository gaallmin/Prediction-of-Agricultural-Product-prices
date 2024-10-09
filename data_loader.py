import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load and split data into training and validation set, to prepare time-series data for predicting avg prices over 3 times units
# return: in nested dictionary set

def data_loader(data_path,  meta_local_path, meta_whole_path, target_items, time_column='시점', year_column='year', month_column='month', soon_column='순',
                feature_columns=None, label_column='평균가격(원)', lookback=9, prediction_window=3, batch_size=32):
    '''
    lookback: # of past time uses as input
    혹시나 싶어서 year, month, soon (순) 다 seperate 하게 column 처리
    '''
    df = pd.read_csv(data_path)  
    meta_local = pd.read_csv(meta_local_path)
    meta_whole = pd.read_csv(meta_whole_path)

    df = pd.merge(df, meta_local[['시점', '품목명', '품종명', '등급명', '평균가(원/kg)','중간가(원/kg)','최저가(원/kg)','최고가(원/kg)', '평년 평균가격(원) Common Year SOON','전순 평균가격(원) PreVious SOON','전달 평균가격(원) PreVious MMonth','전년 평균가격(원) PreVious YeaR']], 
                  how='left', on=['시점', '품목명', '품종명']) # 등급명 넣을까하더라 넣으면 train data랑 달라서 뺐어
    df = pd.merge(df, meta_whole[['시점', '품목명', '품종명', '등급명', '평균가(원/kg)','중간가(원/kg)','최저가(원/kg)','최고가(원/kg)', '평년 평균가격(원) Common Year SOON','전순 평균가격(원) PreVious SOON','전달 평균가격(원) PreVious MMonth','전년 평균가격(원) PreVious YeaR']], 
                  how='left', on=['시점', '품목명', '품종명'], suffixes=('_local', '_whole'))
    

    if feature_columns is None:
        feature_columns = ['품종명', '거래단위', '등급명', '평균가(원/kg)','중간가(원/kg)','최저가(원/kg)','최고가(원/kg)', '평년 평균가격(원) Common Year SOON','전순 평균가격(원) PreVious SOON','전달 평균가격(원) PreVious MMonth','전년 평균가격(원) PreVious YeaR']
    feature_columns.extend([year_column, month_column, soon_column])

    item_batches = {}

    for item in target_items:
        item_data = df[df['품목명'] == item].sort_values(by=[time_column])

        sequences_x = []
        sequences_y = []

        for i in range(len(item_data) - lookback - prediction_window + 1):
            input_seq = item_data[feature_columns].iloc[i:i + lookback].values
            output_seq = item_data[label_column].iloc[i + lookback:i + prediction_window].values
            sequences_x.append(input_seq)
            sequences_y.append(output_seq)
        
        x = np.array(sequences_x)
        y = np.array(sequences_y)
        item_batches[item] = (x,y)
    
    return item_batches