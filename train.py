# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 4:46 下午
# @Author  : jeffery
# @FileName: train.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from utils import WordEmbedding
import torch
import numpy as np
from model import makeModel, makeLoss, makeMetrics, makeOptimizer, makeLrSchedule
from utils import ConfigParser
import yaml
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def active_learning(config):
    from data_process import makeDataSet
    from trainer.trainer import Trainer

    logger = config.get_logger('train')
    train_set,valid_set,query_set = makeDataSet(config)

    model = makeModel(config)
    logger.info(model)

    criterion = makeLoss(config)
    metrics = makeMetrics(config)

    optimizer = makeOptimizer(config, model)
    # lr_scheduler = makeLrSchedule(config, optimizer, train_set)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_dataset=train_set,
                      valid_dataset=valid_set,
                      query_dataset=query_set,
                      test_dataset=None,
                      # lr_scheduler=lr_scheduler
                      )

    trainer.train()

def run(config_fname):
    with open(config_fname, 'r', encoding='utf8') as f:
        config_params = yaml.load(f, Loader=yaml.Loader)
        config_params['config_file_name'] = config_fname

    config = ConfigParser.from_args(config_params)
    active_learning(config)

if __name__ == '__main__':
    run('configs/al_transformers_pure.yml')
