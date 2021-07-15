import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

import os
from time import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from bert_models.base_bert_model import BaseBertModel

class BaseBertModelInterventions(BaseBertModel):
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    def __init__(self, output_dir, label_list, gpu_device_num_hub=0, gpu_device_num = 1, batch_size = 16, max_seq_length = 256,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "", label_column = "label", multilabel=False):
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,\
        bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column, multilabel=multilabel)

    def get_text_a(self, x):
       return x["Narrow concept"]

    def get_text_b(self, x):
       return ""

    def get_text_a_from_tuple(self, x):
        return x[0]

    def get_text_b_from_tuple(self, x):
        return ""