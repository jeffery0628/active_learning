# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 1:47 下午
# @Author  : jeffery
# @FileName: weibo_trainer.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from utils import inf_loop, MetricTracker
from base import BaseTrainer
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, ConcatDataset
import utils.query_strategies as module_query_strategies
import random
import transformers
from typing import List


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_dataset,
                 valid_dataset=None, query_dataset=None, test_dataset=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        data_loader = DataLoader(train_dataset, batch_size=train_dataset.batch_size,
                                 num_workers=train_dataset.num_workers, collate_fn=train_dataset.collate_fn)
        self.train_dataset = train_dataset
        self.query_pool = query_dataset
        self.idxs_labeled = np.zeros(len(query_dataset), dtype=bool) # 用于记录查询集中被标记过的样本
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = None
        self.test_data_loader = None
        if valid_dataset:
            self.valid_data_loader = DataLoader(valid_dataset, batch_size=valid_dataset.batch_size,
                                                num_workers=valid_dataset.num_workers,
                                                collate_fn=valid_dataset.collate_fn)
        if test_dataset:
            self.test_data_loader = DataLoader(test_dataset, batch_size=test_dataset.batch_size,
                                               num_workers=test_dataset.num_workers, collate_fn=test_dataset.collate_fn)

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        data_iter = iter(self.data_loader)
        batch_idx = 0
        while True:
            try:
                data = next(data_iter)
            except StopIteration:
                break

            self.optimizer.zero_grad()
            input_ids, attention_masks, labels, text_lengths = data
            if 'cuda' == self.device.type:
                input_ids = input_ids.cuda()
                attention_masks = attention_masks.cuda()
                text_lengths = text_lengths.cuda()
                labels = labels.cuda()
            preds, embedding = self.model(input_ids, attention_masks, text_lengths)
            loss = self.criterion[0](preds, labels)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            preds = torch.round(torch.sigmoid(preds))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(preds, labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.3f}'.format(epoch, self._progress(batch_idx),
                                                                           loss.item()))

            if batch_idx == self.len_epoch:
                break
            batch_idx += 1
        log = self.train_metrics.result()

        # 验证集
        if self.valid_data_loader:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        # 测试集
        if self.test_data_loader:
            test_log = self._inference_epoch(epoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})

        # 查询集
        if self.not_improved_count > self.early_stop:  # 训练充分之后再进行query
            self._query_epoch()
            self.len_epoch = len(self.data_loader)
            self.not_improved_count = 0
            log.update({
                'num sample in train set': len(self.train_dataset),
                'num sample in query pool': np.sum(~self.idxs_labeled)
            })

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                input_ids, attention_masks, labels, text_lengths = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    text_lengths = text_lengths.cuda()
                    labels = labels.cuda()
                logists, embedding = self.model(input_ids, attention_masks, text_lengths)

                if self.add_graph: # 把网络结构图写到tensorboard上
                    self.writer.writer.add_graph(self.model.module,
                                                 [input_ids, attention_masks, text_lengths])
                    self.add_graph = False

                loss = self.criterion[0](logists, labels)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                preds = torch.round(torch.sigmoid(logists))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(preds, labels))
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _inference_epoch(self, epoch):
        """
        Inference after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                input_ids, attention_masks, labels, text_lengths = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    text_lengths = text_lengths.cuda()
                    labels = labels.cuda()
                logists, embedding = self.model(input_ids, attention_masks, text_lengths)
                loss = self.criterion[0](logists, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'test')
                self.test_metrics.update('loss', loss.item())
                preds = torch.round(torch.sigmoid(logists))
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(preds, labels))

                # add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
            return self.test_metrics.result()

    def _query_epoch(self):
        query_labeled = self._query()
        # self._query_train()
        sample_labeled = np.asarray(self.query_pool)[query_labeled]
        self.train_dataset.features.extend(sample_labeled)
        self.data_loader = DataLoader(self.train_dataset, batch_size=self.train_dataset.batch_size,
                                      num_workers=self.train_dataset.num_workers, shuffle=True,
                                      collate_fn=self.train_dataset.collate_fn)

    def _query(self):
        """
        查询最不确定的样本
        :return:
        """
        query_num = self.config['active_learning']['query_num'] # 对多少个样本进行查询，如果值为None，则对样本池中所有样本进行查询
        if not isinstance(query_num, int):
            query_num = len(self.query_pool)

        idxs_unlabeled = np.arange(len(self.query_pool))[~self.idxs_labeled] # 取出没有被标注的样本下标
        unlabeled_dataloader = DataLoader(np.array(self.query_pool)[idxs_unlabeled],
                                          batch_size=self.query_pool.batch_size,
                                          shuffle=False,
                                          num_workers=self.query_pool.num_workers,
                                          collate_fn=self.query_pool.collate_fn)
        indexes = []
        logists = []
        embeddings = []
        self._resume_checkpoint(self.best_path)  # 加载最好的模型进行查询

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(unlabeled_dataloader):
                if batch_idx * self.query_pool.batch_size >= query_num:
                    break
                input_ids, attention_masks, labels, text_lengths = data
                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    text_lengths = text_lengths.cuda()
                logist, embedding = self.model(input_ids, attention_masks, text_lengths)
                indexes.extend(list(idxs_unlabeled[batch_idx * self.query_pool.batch_size:(
                                                                                                  batch_idx + 1) * self.query_pool.batch_size]))
                logists.append(logist.cpu())
                embeddings.append(embedding.cpu())
        logists = torch.cat(logists, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        indexes = np.asarray(indexes)
        assert logists.shape[0] == len(indexes), 'the number of sample is not equal to its index number'

        # 主动学习，对top_n 个query 进行标注
        query_labeled = self.config.init_ftn('active_learning', module_query_strategies)(logists=logists,
                                                                                         embeddings=embeddings,
                                                                                         idxs_unlabeled=indexes)
        self.idxs_labeled[query_labeled] = True
        return query_labeled

    # def _query_train(self):
    #     idxs_labeled = np.arange(len(self.query_pool))[self.idxs_labeled]
    #     labeled_dataloader = DataLoader(np.array(self.query_pool)[idxs_labeled], batch_size=self.query_pool.batch_size,
    #                                     num_workers=self.query_pool.num_workers, collate_fn=self.query_pool.collate_fn,
    #                                     shuffle=True
    #                                     )
    #     self.model.train()
    #     for batch_idx, data in enumerate(labeled_dataloader):
    #         input_ids, text_lengths, labels = data
    #         if 'cuda' == self.device.type:
    #             input_ids = input_ids.cuda()
    #             text_lengths = text_lengths.cuda()
    #             labels = labels.cuda().long()
    #         logists, embedding = self.model(input_ids, None, text_lengths)
    #         loss = self.criterion[0](logists, labels)
    #         loss.backward()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def makeLrSchedule(self):
        # lr_scheduler = config.init_obj('lr_scheduler', optimization.lr_scheduler, optimizer)
        lr_scheduler = self.config.init_obj('lr_scheduler', transformers.optimization, self.optimizer,
                                            num_training_steps=int(
                                                len(self.train_dataset) / self.train_dataset.batch_size))
        return lr_scheduler
