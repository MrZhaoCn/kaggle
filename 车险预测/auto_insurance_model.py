import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import itertools
import operator
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance

def cleanData():
    train_data = pd.read_csv('./data/train.csv')
    for coloum in train_data.columns:
        # 缺省值用nan替换
        train_data[coloum].replace(-1, np.nan, inplace=True)

    # 对于缺省值过多的特征一般大于80%，一种方式是删除、另一种方式是利用哑编码，原来的特征不参与建模。ps_car_03_cat缺失值大于0.69，用亚编码替换
    # train_data.drop(columns=['ps_car_03_cat'], inplace=True)
    #哑编码
    train_data["ps_car_03_cat"] = train_data["ps_car_03_cat"].isna().apply(int)

    # 用均值填充 nan，但注意，对于二值的或者分类的特征用均值不合适，这种类型的可以不处理，把缺省值当成一类，连续的数据合适
    # mean_cols = train_data.mean()
    # train_data.fillna(mean_cols, inplace=True)
    for coloum in train_data.columns:
        if 'bin' in coloum or 'cat' in coloum:
            # 对于二值的或者分类的特征用用出现最多的数值替换、或者不处理、缺失值当成一类
            continue
        else:
            # 用均值填充
            train_data[coloum].fillna(train_data[coloum].mean(), inplace=True)
    # 查看每个特征nan所占的比例
    #print(train_data.apply(lambda col: sum(col.isna()) / col.size))

    #查看方差，对于方差很小的特征（二值或者分类特征除外），可以去除
    # for coloum in train_data.columns:
    #     print(train_data[coloum].describe())
    train_data.to_csv("./data/data_handle.csv", index=False)

def getData():
    train_data = pd.read_csv('./data/data_handle.csv')
    y_train = train_data["target"]
    x_train = train_data.drop(columns=['target', 'id'])

    #对数据特征进行标准化（二值和类别行的除外）
    for coloum in x_train.columns:
        if 'bin' in coloum or 'cat' in coloum:
            continue
        else:
            # 用均值填充
            x_train[coloum] = (x_train[coloum] - x_train[coloum].min()) / (x_train[coloum].max() - x_train[coloum].min())
    # x_train.to_csv("./data/x_train.csv", index=False)
    #画图查看各特征与目标之间是否存在某种分布关系，若没明显分布关系可去除

    # 把dataframe的值取出转成数组
    return x_train.values, y_train

# cleanData()
x_train, y_train = getData()
#网格搜索寻找最优参数
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
max_depth = [2, 4, 6, 8, 10, 12]
model = XGBClassifier()
param_grid = dict(learning_rate=learning_rate,max_depth=max_depth)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#画出最优的特征
plot_importance(model)
pyplot.show()







