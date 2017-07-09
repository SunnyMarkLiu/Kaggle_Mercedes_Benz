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

other_xgb_model_result = pd.read_csv('other_xgb_model_result.csv')  # 0.56784
stacked_models_result = pd.read_csv('stacked-models.csv')           # 0.56894
xgboost_result = pd.read_csv('xgboost_submission.csv')              # 0.5637

result = pd.DataFrame({'ID': xgboost_result['ID']})
result['y'] = 0.55 * stacked_models_result['y'] + 0.3 * other_xgb_model_result['y'] + 0.15 * xgboost_result['y']

result.to_csv(Configure.submission_path, index=False)
