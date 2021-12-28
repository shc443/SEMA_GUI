# Copyright Xinapse NLU team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import pytorch_lightning as pl
import pickle
from collections import Counter


class VOC_Dataset2(Dataset):
    "Transform SEMA VOC data into encoded tensor"

    def __init__(self, data, tokenizer, max_token_len: int = 512):
        """
        Args:
            :param data: pandas dataframe, simply VOC excel data from SEMA
            :param tokenizer: trained tokenizer from KLUE RoBERTa
            :param max_token_len: maximum length of tensor for each input tensor
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        voc_text = data_row.VOC
        voc_labels = data_row.label

        encoding = self.tokenizer.encode_plus(voc_text,
                                              add_special_tokens=True,
                                              max_length=self.max_token_len,
                                              return_token_type_ids=False,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt')

        return dict(voc_text=voc_text,
                    input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(),
                    labels=torch.FloatTensor(voc_labels))


class VOC_DataModule(pl.LightningDataModule):
    '''
    Data Module for SEMA VOC classification task.
    train, val, test splits and transforms
    '''

    def __init__(self, train_df, test_df, tokenizer, batch_size: int = 4, max_token_len: int = 200):
        """
        Args:
            :param train_df: train dataset
            :param test_df: validation dataset
            :param tokenizer: trained tokenizer from KLUE RoBERTa
            :param batch_size:  desired batch size
            :param max_token_len: maximum length of tensor for each input tensor
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.test_dataset = VOC_Dataset2(self.test_df, self.tokenizer, self.max_token_len)

    def setup(self, stage=None):
        """Encode the train & valid dataset (already splitted)"""
        self.train_dataset = VOC_Dataset2(self.train_df, self.tokenizer, self.max_token_len)
        self.test_dataset = VOC_Dataset2(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        """VOC train set removes a subset to use for validation"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """VOC validation set removes a subset to use for validation"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """VOC test set"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class VOC_Data_Transform:
    # directory = os.path.abspath(os.getcwd())
    # input_files = pd.Series(
    #     [f[:-5] for f in listdir(directory + '/voc_data') if isfile(join(directory + '/voc_data', f))])
    # output_files = [f[:-5] for f in listdir(directory + '/output') if isfile(join(directory + '/output', f))]
    # running_files = input_files[~(input_files + '_output').isin(output_files)]
    def __init__(self, file: str = None):
        self.directory = os.path.abspath(os.getcwd())
        self.file = file
        self.voc_etc2 = pd.read_pickle(self.directory + '/pickles/voc_etc2.pkl')
        self.keyword = pd.read_pickle(self.directory + '/pickles/keyword.pkl')
        self.tokenizer = pickle.load(open(self.directory + "/pickles/tokenizer.pkl", "rb"))


    def transform(self):
        # input files
        voc_testset = pd.read_excel(self.file, dtype=str)
        voc_testset['VOC1'] = voc_testset.VOC1.str.replace('\n', ' ')
        voc_testset['VOC2'] = voc_testset.VOC2.str.replace('\n', ' ')

        voc = pd.concat([voc_testset.VOC1, voc_testset.VOC2]).sort_index().values
        voc_testset = pd.concat([voc_testset] * 2).sort_index().iloc[:, :-2]
        voc_testset['VOC'] = voc
        voc_testset['VOC'].fillna('', inplace=True)
        voc_testset['VOC'] = voc_testset['VOC'].apply(str)
        voc_testset.reset_index(inplace=True)
        voc_testset['label'] = pd.DataFrame(np.zeros((16, voc_testset.shape[0])).T).astype(int).apply(
            list, axis=1)
        return voc_testset

    def filter_etc2(self, df):
        voc_col = df['VOC'].apply(lambda x: re.sub('[^A-Za-z0-9가-힣 ]', '', x))
        filt0 = (voc_col.str.len() < 2).astype(int)
        filt1 = voc_col.apply(lambda x: bool(re.match(r'^[_\W]+$', str(x).replace(' ', '')))).astype(int)
        filt2 = voc_col.apply(lambda x: bool(re.match(r'[\d/-]+$', str(x).replace(' ', '')))).astype(int)
        filt3 = voc_col.str.replace(' ', '').str.split('').fillna('').apply(set).str.len() == 2
        voc_col_enc = voc_col.apply(lambda x: Counter(self.tokenizer.encode_plus(x,
                                                                            add_special_tokens=True,
                                                                            max_length=200,
                                                                            return_token_type_ids=False,
                                                                            truncation=True,
                                                                            return_attention_mask=True,
                                                                            return_tensors='pt')['input_ids'].numpy()[
                                                          0][1:-1]).keys())
        filt4 = voc_col_enc.apply(lambda x: ','.join([str(y) for y in x])).isin(
            self.voc_etc2.apply(lambda x: ','.join([str(y) for y in x.keys()])))
        return filt0 + filt1 + filt2 + filt3 + filt4

    def find_keyword(self, df):
        def findall_vec(key, voc):
            try:
                return re.findall(key, voc)[0]
            except:
                return ''

        def findall_vec2(df):
            return findall_vec(df['keyword'], df['VOC'])

        df['topic'] = df.pred.str.split('_').str[0]
        df['sentiment'] = df.pred.str.split('_').str[1]
        df.topic.fillna('기타', inplace=True)
        df['keyword'] = self.keyword.loc[df.topic].values
        df['keyword'] = df.apply(findall_vec2, axis=1)
        return df
