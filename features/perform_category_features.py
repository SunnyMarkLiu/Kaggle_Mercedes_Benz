#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-1 下午1:35
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
# remove warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn import preprocessing
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    print 'numerical encoder...'
    str_columns = train.select_dtypes(include=['object']).columns
    for c in str_columns:
        df2 = pd.DataFrame({col: vals['y'] for col, vals in train.groupby(c)})
        sorded_df = df2.median().sort_values(ascending=True)
        c_mean = sorded_df[c].mean()

        feature_map = {k: v + 1 for v, k in enumerate(sorded_df.index.values)}
        train['numerical_' + c] = train[c].map(feature_map)
        test['numerical_' + c] = test[c].map(feature_map)
        if c not in feature_map.keys():
            train['numerical_' + c] = c_mean
            test['numerical_' + c] = c_mean

    train_y = train['y']
    train.drop(['y'], axis=1, inplace=True)
    # 合并训练集和测试集
    conbined_data = pd.concat([train, test])
    ids = conbined_data['ID']

    label_encoder_df = pd.DataFrame({'ID': ids})
    print 'perform label encoder...'
    for c in str_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(conbined_data[c].values))
        label_encoder_df['label_encoder_' + c] = lbl.transform(list(conbined_data[c].values))

    # print 'perform dummy encoder...'
    # dummy_encoder_df = pd.DataFrame({'ID': ids})
    # for c in str_columns:
    #     dummies_df = pd.get_dummies(conbined_data[c], prefix=c)
    #     dummy_encoder_df = pd.concat([dummy_encoder_df, dummies_df], axis=1)

    # 合并数据
    del label_encoder_df['ID']
    conbined_data = pd.concat([conbined_data, label_encoder_df], axis=1)
    # del dummy_encoder_df['ID']
    # conbined_data = pd.concat([conbined_data, dummy_encoder_df], axis=1)

    # 去除原有的 category features
    conbined_data.drop(str_columns, axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['y'] = train_y.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform category features =========='
    main()
