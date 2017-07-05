#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-5 下午4:16
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
# remove warnings
import warnings

warnings.filterwarnings('ignore')

import cPickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from conf.configure import Configure
from utils import data_util


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    print 'train:', train.shape, ', test:', test.shape

    train_y = train['y']
    if (not os.path.exists(Configure.kmeans_feature_distance_train_path)) or \
            (not os.path.exists(Configure.kmeans_feature_distance_test_path)):
        train.drop(['y'], axis=1, inplace=True)
        # 合并训练集和测试集
        conbined_data = pd.concat([train, test])
        ids = conbined_data['ID']
        results_df = pd.DataFrame({'ID': ids})

        n_clusters = 4
        cls = KMeans(n_clusters=n_clusters, n_jobs=-1)
        kmeans_labels = cls.fit_predict(conbined_data.values)
        conbined_data['cluster_label'] = kmeans_labels
        results_df['cluster_label'] = kmeans_labels

        # 计算距离
        cluster_centers = cls.cluster_centers_

        def calc_intra_center_distance(data):
            center = cluster_centers[int(data['cluster_label']), :]
            raw_data = data.drop(['cluster_label'])
            return np.linalg.norm(center - raw_data)

        def calc_extra_center_distance(data):
            ci = range(0, n_clusters)
            ci.remove(int(data['cluster_label']))

            distance = 0.0
            raw_data = data.drop(['cluster_label'])
            for i in ci:
                center = cluster_centers[i, :]
                dis = np.linalg.norm(center - raw_data)
                distance += dis
            return distance

        def calc_extra_center_distance2(data):
            ci = range(0, n_clusters)
            ci.remove(int(data['cluster_label']))

            distance = 0.0
            raw_data = data.drop(['cluster_label'])
            for i in ci:
                center = cluster_centers[i, :]
                dis = np.linalg.norm(center - raw_data)
                distance += dis
            return distance / len(ci)

        print 'calc intra center distance...'
        results_df['intra_center_distance'] = conbined_data.apply(calc_intra_center_distance, axis=1, raw=True)
        print 'calc average extra center distance...'
        results_df['average_extra_center_distance'] = conbined_data.apply(calc_extra_center_distance, axis=1, raw=True)
        print 'calc sum extra center distance...'
        results_df['sum_extra_center_distance'] = conbined_data.apply(calc_extra_center_distance2, axis=1, raw=True)

        center_distance_train = results_df.iloc[:train.shape[0], :]
        center_distance_test = results_df.iloc[train.shape[0]:, :]

        with open(Configure.kmeans_feature_distance_train_path, "wb") as f:
            cPickle.dump(center_distance_train, f, -1)
        with open(Configure.kmeans_feature_distance_test_path, "wb") as f:
            cPickle.dump(center_distance_test, f, -1)
    else:
        with open(Configure.kmeans_feature_distance_train_path, "rb") as f:
            center_distance_train = cPickle.load(f)
        with open(Configure.kmeans_feature_distance_test_path, "rb") as f:
            center_distance_test = cPickle.load(f)

    # merge
    train = pd.merge(train, center_distance_train, how='left', on='ID')
    test = pd.merge(test, center_distance_test, how='left', on='ID')

    train['y'] = train_y.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== generate feature distance =========='
    main()
