import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import Levenshtein
df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
df_desc = pd.read_csv('./input/product_descriptions.csv')

#合并测试/训练集，以便于统一做进一步的文本预处理
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=False)
#产品介绍也是很有用的信息，合并
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

#词干提取
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

#统计关键词出现次数
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

#计算搜索词和标题的距离
df_all['dist_in_title']= df_all.apply(lambda x:Levenshtein.ratio(x['search_term'], x['product_title']), axis=1)
df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'], x['product_description']), axis=1)



