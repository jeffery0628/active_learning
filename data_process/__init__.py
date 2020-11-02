# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:08 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from data_process import data_process as module_data_process


def makeDataSet(config):
    # setup data_set, data_process instances
    train_set = config.init_obj('train_set', module_data_process)
    valid_set = config.init_obj('valid_set', module_data_process)
    query_set = config.init_obj('query_set', module_data_process)

    return train_set, valid_set, query_set
