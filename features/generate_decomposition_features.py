#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-29 下午2:21
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from conf.configure import Configure


def main():
    print 'load datas...'
    train = pd.read_csv(Configure.original_train_path)
    test = pd.read_csv(Configure.original_test_path)

if __name__ == '__main__':
    main()
