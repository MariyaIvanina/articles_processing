from outcomes_modelling.outcomes_sentence_labeler import OutcomesSentenceLabeler

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
import spacy

nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

class OutcomesSentenceLabelerPlusNer(OutcomesSentenceLabeler):

    def __init__(self, output_dir, label_list = [0, 1],
            gpu_device_num_hub = 0, gpu_device_num = 1, batch_size = 32, max_seq_length = 128,
            bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "",
            label_column = "label", use_concat_results = False, path_to_NER="", threshold=0.5):
        OutcomesSentenceLabeler.__init__(self, output_dir, label_list, gpu_device_num_hub=gpu_device_num_hub, 
            gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,
            bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column,
            use_concat_results=use_concat_results
            )
        self.path_to_NER = path_to_NER
        self.ner_model = spacy.load(path_to_NER)
        self.threshold = threshold

    def predict_for_sentences(self, sentences_data, only_res=True):
        sentence_parts2id = {}
        global_id = 0
        data_for_predict = []
        for i in range(len(sentences_data)):
            sentence_parts2id[i] = []
            for ent in self.ner_model(sentences_data[i][0]).ents:
                sentence_parts2id[i].append(global_id)
                data_for_predict.append((ent.text, -1))
                global_id += 1
        res_full_prob, res_full_pred_label, res_full_true = self.predict_for_df(
            pd.DataFrame(data_for_predict, columns=["text", "label"]))
        if only_res:
            res = []
            for i in range(len(sentence_parts2id)):
                full_result = 0
                for j in sentence_parts2id[i]:
                    if full_result != 1:
                        full_result = (res_full_prob[j][1] >= self.threshold)
                res.append(full_result)
        else:
            res = []
            for i in range(len(sentence_parts2id)):
                full_result = 0
                outcomes_extracted_results = []
                for j in sentence_parts2id[i]:
                    label_ = (res_full_prob[j][1] >= self.threshold)
                    outcomes_extracted_results.append((data_for_predict[j][0], label_))
                    if full_result != 1:
                        full_result = label_
                res.append((full_result, outcomes_extracted_results))
        return res

    def evaluate_model(self, test, is_head = True):
        res = self.predict_for_sentences(test.values)
        res_label = []
        if type(test) == list:
            for i in range(len(test)):
                res_label.append(test[i][1])
        else:
            res_label = list(test[self.label_column].values)
        self.print_summary(res_label, res)
        return res

    def train_model(self, train, test=[]):
        print("The class is not for training")
