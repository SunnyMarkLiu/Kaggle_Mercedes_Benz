#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
basic xgboost baseline with basic original features
@author: MarkLiu
@time  : 17-6-29 上午9:33
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# my own module
from conf.configure import Configure


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


def main():
    print 'load datas...'
    train = pd.read_csv(Configure.original_train_path)
    test = pd.read_csv(Configure.original_test_path)

    str_columns = train.select_dtypes(include=['object']).columns

    for c in str_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

    y_train_all = train['y']
    del train['ID']
    del train['y']
    id_test = test['ID']
    del test['ID']

    test_size = (1.0 * test.shape[0]) / train.shape[0]
    print "submit test size:", test_size

    # Convert to numpy values
    X_all = train.values
    # Create a validation set, with last 20% of data
    num_train = int(train.shape[0] * 0.8)
    X_train_all = X_all
    X_train = X_all[:num_train]
    X_val = X_all[num_train:]
    y_train = y_train_all[:num_train]
    y_val = y_train_all[num_train:]
    X_test = test
    print "validate size:", 1.0 * X_val.shape[0] / X_train.shape[0]

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', y_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', y_val.shape)
    print('X_test shape is', X_test.shape)

    df_columns = train.columns.values
    dtrain_all = xgb.DMatrix(X_train_all, y_train_all, feature_names=df_columns)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    num_round = 1000
    xgb_params['nthread'] = 24
    evallist = [(dval, 'eval')]

    bst = xgb.train(xgb_params, dtrain, num_round, evallist, early_stopping_rounds=10,
                    verbose_eval=10)

    train_rmse = mean_squared_error(y_train, bst.predict(dtrain))
    val_rmse = mean_squared_error(y_val, bst.predict(dval))
    print 'train_rmse =', np.sqrt(train_rmse), ', val_rmse =', np.sqrt(val_rmse)

    num_boost_round = bst.best_iteration
    print 'best_iteration: ', num_boost_round
    model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

    print 'predict submit...'
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    main()
