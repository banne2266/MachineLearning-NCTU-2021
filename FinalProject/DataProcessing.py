import pandas as pd
import glob
import os
import pandas as pd
import numpy as np

def read_data(path:str):
    if path[-4:] == '.csv':
        return pd.read_csv(path)
    
    DataFrames = []
    files = glob.glob(os.path.join(path, "*"))
    for file in files:
        DataFrames.append(pd.read_csv(file))
    return pd.concat(DataFrames, ignore_index=True)

    
def preprocess_drop_col(data: pd.DataFrame):
    data = data.drop('車輛撞擊部位其他', 1)
    data = data.drop('年', 1)
    data = data.drop('月', 1)
    data = data.drop('日', 1)
    data = data.drop('分', 1)
    data = data.drop('縣市', 1)
    data = data.fillna(999)
    return data


def preprocess_one_hot(data: pd.DataFrame):
    data_dum = pd.get_dummies(data, columns=['區', '天候', '光線', '事故位置','道路類別', '速限', '道路型態', '事故位置', '路面鋪裝', '路面狀態', '路面缺陷', '障礙物', '視距', '號誌種類', '號誌動作', '分向設施', \
        '快車道或一般車道間', '快慢車道間', '路面邊線', '事故類型及型態', '主要肇因', '受傷程度', '主要傷處', '保護裝備', '行動電話', '當事者區分', '車輛用途', '當事者行動狀態', '駕駛資格情形', \
        '駕駛執照種類', '飲酒情形', '車輛撞擊部位最初', '肇事因素個別', '肇事因素主要', '肇事逃逸', '職業', '旅次目的', '車種', '事故類別'], drop_first=True)
    pd.DataFrame(data_dum)
    return data_dum

def get_label(data: pd.DataFrame):
    label = np.zeros(data.shape[0], np.float)
    for i, element in enumerate(data['受傷']):
        if element > 0:
            label[i] = 1.0
    for i, element in enumerate(data['死']):
        if element > 0:
            label[i] = 1.0
    for i, element in enumerate(data['2-30']):
        if element > 0:
            label[i] = 1.0
    data = data.drop('死', 1)
    data = data.drop('受傷', 1)
    data = data.drop('2-30', 1)
    return data.to_numpy().astype(np.float), label

def process_hour(x):
    return x / 24
def process_longtitute(x):
    return x - 120
def process_latitude(x):
    return x - 23.5

