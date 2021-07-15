from bert_models.base_bert_model import BaseBertModel

from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import re
from text_processing import text_normalizer
import nltk
import os
from time import time
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import os
from study_design_type.study_type_labels import StudyTypeLabels
from utilities import excel_writer
from utilities import excel_reader
import pandas as pd

nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

class OutcomesExtractedLabeler(BaseBertModel):

    def __init__(self, output_dir, label_list = [0, 1],
            gpu_device_num_hub = 0, gpu_device_num = 1, batch_size = 32, max_seq_length = 128,
            bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "",
            label_column = "label", use_concat_results = False, multilabel=False, epoch_num=3):
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num_hub=gpu_device_num_hub, 
            gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,
            bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column, use_concat_results=use_concat_results,
            multilabel=multilabel, epoch_num=epoch_num)

    def prepare_data_df(self, outcomes_train_label_data, label_to_train, proportion_to_check=-1):
        train_data = {}
        real_sentence = {}
        for t in outcomes_train_label_data:
            text = t[0].strip().lower()
            real_sentence[text] = t[0]
            if text not in train_data:
                train_data[text] = 0
            if len(t) > 1:
                if train_data[text] != 1:
                    train_data[text] = int(t[1] == label_to_train)
        data_to_use = [(real_sentence[t], train_data[t]) for t in train_data]
        if proportion_to_check > 0:
            data_use_one = [t for t in data_to_use if t[1] == 1]
            data_use_zero = [t for t in data_to_use if t[1] == 0]
            size_to_check_all = int(len(data_use_one) / proportion_to_check)
            size_to_check_zero = size_to_check_all - len(data_use_one)
            np.random.seed(1237)
            np.random.shuffle(data_use_zero)
            data_to_use = data_use_one + data_use_zero[:size_to_check_zero]
        return pd.DataFrame(data_to_use, columns=["text", "label"]).sample(frac=1)

    def get_text_a(self, x):
       return x["text"]

    def get_text_b(self, x):
       return ""

    def get_text_a_from_tuple(self, x):
        return x[0]

    def get_text_b_from_tuple(self, x):
        return ""