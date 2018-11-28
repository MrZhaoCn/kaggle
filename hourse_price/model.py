import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
train_df = pd.read_csv('./input/train.csv', index_col=0)
test_df = pd.read_csv('./input/test.csv', index_col=0)
#lable本身不平滑，进行平滑后更易训练
y_train = np.log1p(train_df.pop('SalePrice'))

#合并数据,colums方向合并
all_df = pd.concat((train_df, test_df), axis=0)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#print (pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())

#对于数值型的数据进行标准化有利于回归模型的训练
all_dummy_df = pd.get_dummies(all_df)
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
#把dataframe的值取出转成数组
X_train = dummy_train_df.values
X_test = dummy_test_df.values

#5的时候最好
params = [1, 2, 3, 4, 5, 6]
test_scores = []
# for param in params:
#     clf = XGBRegressor(max_depth=param)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
clf = XGBRegressor(max_depth=5)
clf.fit(X_train, y_train)
y_final = np.expm1(clf.predict(X_test))
# plt.plot(params, test_scores)
# plt.title("max_depth vs CV Error")
# plt.show()
submission_df = pd.DataFrame(data={'Id': test_df.index, 'SalePrice': y_final})
submission_df.to_csv("./input/submisson.csv", index=False)
print (submission_df.head())

