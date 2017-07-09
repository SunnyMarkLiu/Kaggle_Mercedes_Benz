#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午4:47
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV, ElasticNet

from model_stack.model_wrapper import XgbWrapper, SklearnWrapper
from model_stack.model_stack import TwoLevelModelStacking

# my own module
from utils import data_util
from conf.configure import Configure

print 'load datas...'
train, test = data_util.load_dataset()
y_train_all = train['y']
del train['ID']
del train['y']
id_test = test['ID']
del test['ID']
print 'train:', train.shape, ', test:', test.shape
random_state = 420

et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

rd_params = {
    'alpha': 10
}

ls_params = {
    'alpha': 0.005
}

elasticnet_param = {
    'l1_ratio': [.1, .5, .7, .9, .95, .99, .995, 1],
    'eps': 0.001,
    'n_alphas': 100, 'fit_intercept': True,
    'normalize': True, 'precompute': 'auto', 'max_iter': 2000, 'tol': 0.0001, 'cv': 5,
    'copy_X': True, 'verbose': 0, 'n_jobs': -1, 'positive': False, 'random_state': 420,
    'selection': 'cyclic'
}
SEED = 0

xgb1 = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
xgb2 = XgbWrapper(seed=SEED, params=xgb_params)

level_1_models = [xgb1, et, rf, rd, ls, xgb2]
stacking_model = SklearnWrapper(clf=ElasticNetCV, seed=SEED, params=elasticnet_param)

model_stack = TwoLevelModelStacking(train, y_train_all, test, level_1_models,
                                    stacking_model=stacking_model,
                                    stacking_with_pre_features=False)
predicts = model_stack.run_stack_predict()

df_sub = pd.DataFrame({'id': id_test, 'y': predicts})
df_sub.to_csv(Configure.submission_path, index=False)
