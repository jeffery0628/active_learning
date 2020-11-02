# -*- coding: utf-8 -*-
# @Time    : 2020/10/16 1:06 下午
# @Author  : jeffery
# @FileName: query_strategies.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch.nn.functional as F
import torch
import numpy as np


def margin_sampling(top_n, logists, embeddings, idxs_unlabeled):
    preds = F.softmax(logists, dim=-1)
    preds_sorted, idxs = preds.sort(descending=True)  # 对类别进行排序
    U = preds_sorted[:, 0] - preds_sorted[:, 1]  # 计算概率最高的前两个类别之前的差值，
    residule_sorted, idx_sorted = U.sort()  # 按照差值升序排序，差值越小说明预测效果越差
    return idxs_unlabeled[idx_sorted[:top_n]]

def random_sampling(top_n, logists, embeddings, idxs_unlabeled):
    """
    随机采样
    :param top_n:
    :param logists:
    :param embeddings:
    :param idxs_unlabeled:
    :return:
    """
    return idxs_unlabeled[:top_n]


def multilabel_margin_sampling(top_n, logists, embeddings, idxs_unlabeled):
    """
    多标签 的不确定性样本选择策略,比如一个样本(6个类别)经过sigmoid之后：preds为[0.1,0.3,0.5,0.1,0.8,0.74]
    选择策略：对于每个样本计算每个类别预测值与0.5的距离之和，对距离之和排序，距离之和越小说明该样本越不确定
    即：
    |0.1-0.5|+|0.3-0.5|+|0.5-0.5|+|0.1-0.5|+|0.8-0.5|+|0.74-0.5|

    :param top_n:
    :param logists:
    :param embedding:
    :param idx_unlabeled:
    :return:
    """
    preds = F.sigmoid(logists)
    uncertainty = torch.sum(torch.abs(preds - torch.ones_like(preds) * 0.5), dim=-1)
    uncertainty_sorted, idx_sorted = uncertainty.sort()  # 与0.5的距离之和越小则越不确定
    return idxs_unlabeled[idx_sorted[:top_n]]
