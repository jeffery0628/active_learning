# -*- coding: utf-8 -*-
# @Time    : 2020/11/2 11:24 上午
# @Author  : jeffery
# @FileName: data_process.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:

from base import BaseDataSet
from pathlib import Path
import pandas as pd
from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import json
import numpy as np
from tqdm import tqdm


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: Optional[str]
    text: str
    label: List[int]

    def __post_init__(self):
        self.text = self.text[:510]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    # token_type_ids: Optional[List[int]] = None
    label: List[int]
    sent_len: int


class MedicalDataset(BaseDataSet):
    def __init__(self, data_dir, file_name, shuffle, transformer_model, overwrite_cache, force_download, cache_dir,
                 batch_size, num_workers):

        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        # 数据模式：训练集/验证集/测试集
        self.feature_cache_file = self.data_dir / '.cache' / '{}.cache'.format(file_name.split('.')[0])
        super(MedicalDataset, self).__init__(transformer_model=transformer_model, overwrite_cache=overwrite_cache,
                                             force_download=force_download, cache_dir=cache_dir)

    def read_examples_from_file(self):
        input_file = self.data_dir / self.file_name
        with input_file.open('r') as f:
            for line in tqdm(f):
                json_line = json.loads(line)
                yield InputExample(guid=json_line['id'], text=json_line['text'], label=json_line['labels'])

    def convert_examples_to_features(self):
        features = []
        for example in self.read_examples_from_file():
            inputs = self.tokenizer.encode_plus(example.text, return_length=True, return_attention_mask=True)
            features.append(InputFeatures(input_ids=inputs.data['input_ids'], attention_mask=inputs.data['attention_mask'], sent_len=inputs.data['length'],
                                          label=example.label))
        return features

    def collate_fn(self, datas):
        max_len = max([data.sent_len for data in datas])

        input_ids = []
        attention_masks = []
        labels = []
        text_lengths = []

        for data in datas:
            input_ids.append(data.input_ids + [self.tokenizer.pad_token_id] * (max_len - data.sent_len))
            attention_masks.append(data.attention_mask + [0] * (max_len - data.sent_len))
            labels.append(data.label)
            text_lengths.append(data.sent_len)

        input_ids = torch.LongTensor(np.array(input_ids))
        attention_masks = torch.LongTensor(np.array(attention_masks))
        labels = torch.FloatTensor(np.array(labels))
        text_lengths = torch.LongTensor(text_lengths)

        return input_ids, attention_masks, labels,text_lengths
