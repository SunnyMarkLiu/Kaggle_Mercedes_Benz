#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-8 下午5:09
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from conf.configure import Configure

elasticnet_result = pd.read_csv('elasticnet_model_result.csv')  # 0.55828
lassolars_result = pd.read_csv('lassolars_model_result.csv')    # 0.55827
rf_result = pd.read_csv('rf_regressor_model_result.csv')        # 0.55199
ridge_result = pd.read_csv('ridge_model_result.csv')            # 0.53696
xgboost_result = pd.read_csv('xgboost_submission.csv')          # 0.5637

result = pd.DataFrame({'ID': elasticnet_result['ID']})
result['y'] = 0.4 * xgboost_result['y'] + 0.3 * elasticnet_result['y'] + 0.15 * lassolars_result['y'] + \
              0.1 * rf_result['y'] + 0.05 * ridge_result['y']

result.to_csv(Configure.submission_path, index=False)
