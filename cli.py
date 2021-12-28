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
import re
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import time
import datetime
import pickle
from collections import Counter
import pytorch_lightning as pl
from data import VOC_DataModule, VOC_Data_Transform
from model import VOC_TopicLabeler
import torch.nn as nn
from __init__ import cli_VOC_logo
import logging
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SEMA:
    def __init__(self,
                 N_EPOCHS: int = 10,
                 BATCH_SIZE: int = 12,
                 MAX_LEN: int = 256,
                 LR: float = 2e-05,
                 opt_thresh: float = 0.4,
                 file_path = None):
        """
        Args:
            :param N_EPOCHS: Number of Epochs
            :param BATCH_SIZE: Number of Batch Size
            :param MAX_LEN: Maximum length of Padding
            :param LR: Learning Rate
            :param opt_thresh: Optinmal Threshold for logit classification
            :file_path: Path for file to process (only for gui)
        """
        
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%I:%M:%S %p', level=logging.INFO)
        self.logger = logging.getLogger('sema_logger')

        self.N_EPOCHS = N_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LEN = MAX_LEN
        self.LR = LR
        self.opt_thresh = opt_thresh
        self.file_path = file_path

        self.directory = os.path.abspath(os.getcwd())
        self.config = pickle.load(open(self.directory + "/pickles/config.pkl", "rb"))
        self.tokenizer = pickle.load(open(self.directory + "/pickles/tokenizer.pkl", "rb"))

        with open(self.directory + '/pickles/data.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
        LABEL_COLUMNS = self.mlb.classes_[:]

        self.new_model = VOC_TopicLabeler.load_from_checkpoint(
            checkpoint_path=self.directory + "/model_weights/hosrevroberta_210825_5.ckpt",
            n_classes=len(LABEL_COLUMNS),
            config=self.config)
        self.new_model.eval()

    def find_file_list(self):
        """Finding files for inference
            Returns:
                running_files: list, list of file paths to be proceed as string
        """
        self.logger.info('Loading Working Files from voc_data folder...')
        input_files = pd.Series(
            [f[:-5] for f in listdir(self.directory + '/voc_data') if isfile(join(self.directory + '/voc_data', f))])
        output_files = [f[:-5] for f in listdir(self.directory + '/output') if
                        isfile(join(self.directory + '/output', f))]
        running_files = input_files[~(input_files + '_output').isin(output_files)]
        return running_files

    def trainer_setup(self, voc_testset):
        """Setting up data module & trainer for inference
            Args:
                :param voc_testset: dataframe from inference dataset

            Returns:
                None: just update voc_testset
        """
        self.logger.info('Load Data Module & Trainer for inference...')
        self.voc_testset = voc_testset
        self.data_module = VOC_DataModule(self.voc_testset,
                                          self.voc_testset,
                                          tokenizer=self.tokenizer,
                                          batch_size=self.BATCH_SIZE,
                                          max_token_len=self.MAX_LEN)
        self.data_module.setup()
        #   accelerator = CPUAccelerator(training_type_plugin=DDPPlugin(),precision_plugin=NativeMixedPrecisionPlugin())
        #   trainer = pl.Trainer(accelerator=accelerator,max_epochs=N_EPOCHS, progress_bar_refresh_rate=3)
        self.trainer = pl.Trainer(max_epochs=self.N_EPOCHS, progress_bar_refresh_rate=1)

    def inference(self):
        """Inferencing the dataset"""
        self.logger.info('Inferencing the dataset...')
        testing_predict = self.trainer.predict(self.new_model, datamodule=self.data_module)
#        sema_df_final = np.vstack(
#            pd.Series(np.vstack(testing_predict)[:, 1]).apply(lambda x: np.vstack(x.detach().cpu().clone().numpy()))) #only for gpus
        sema_df_final = np.vstack([batch[1] for batch in testing_predict])
        pred_label = (sema_df_final > self.opt_thresh).astype(int)
        self.voc_testset['pred'] = pd.Series(self.mlb.inverse_transform(pred_label)).apply(list)
        del self.voc_testset['label']

    def VOC_filter_etc(self):
        """Filtering ETC topics based on labelled etc dataset"""
        self.logger.info('Filtering ETC from the result...')
        testing = VOC_Data_Transform().filter_etc2(self.voc_testset)
        self.voc_testset.pred.loc[testing > 0] = [[] for _ in range((testing > 0).sum())]
        self.voc_testset = self.voc_testset.explode('pred', ignore_index=True)

    def VOC_find_keyword(self):
        self.logger.info('Keyword Searching...')
        """Finding keywords from given labelled keyword lists for each topic"""
        self.voc_testset = VOC_Data_Transform().find_keyword(self.voc_testset)

    def save_output(self, file: str = None):
        """Saving Outputs in Excel"""
        self.logger.info('Saving the Final Result...')
        engines = ['openpyxl', 'xlsxwriter']
        for engine in engines:
            # noinspection PyBroadException
#            try:
            self.voc_testset.fillna('').astype(str).to_excel(
                self.directory + '/output/' + file + '_output.xlsx',
                encoding='utf-8-sig', engine=engine)
#            except:
#                continue

    def process_analysis(self):
        "Run all the methods for all files"
        running_files = self.find_file_list()
        for file in running_files[~running_files.str.startswith('.')]:
            self.logger.info('Working on ' + file + '...')
            voc_testset = VOC_Data_Transform(file=file).transform()
            self.trainer_setup(voc_testset)
            self.inference()
            self.VOC_filter_etc()
            self.VOC_find_keyword()
            self.save_output(file=file)

        self.logger.info('DONE. Please check the output folder.')
    
    def process_analysis_gui(self):
        "Run all the methods for all files"
        self.logger.info('Working on ' + self.file_path + '...')
        voc_testset = VOC_Data_Transform(file=self.file_path).transform()
        self.trainer_setup(voc_testset)
        self.inference()
        self.VOC_filter_etc()
        self.VOC_find_keyword()
        self.save_output(file=self.file_path)

        self.logger.info('DONE. Please check the output folder.')
        
if __name__ == "__main__":
    cli_VOC_logo('sema')
#    SEMA().process_analysis()
    SEMA().process_analysis_gui()
    cli_VOC_logo('xinapse')
    
    
