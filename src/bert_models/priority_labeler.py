from bert_models.base_bert_model_with_boosting import BaseBertModelWithBoost
from text_processing import text_normalizer
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

class PrioirityLabeler(BaseBertModelWithBoost):

    def __init__(self, output_dir, label_list = [0,1], boost_model = "", gpu_device_num = 1, batch_size = 16, max_seq_length = 256,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "../scibert_scivocab_uncased",  label_column = "label"):
        BaseBertModelWithBoost.__init__(self, output_dir, label_list, boost_model = boost_model, gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,\
        bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column)
        self.n_estimators = 50
        self.max_depth = 5
        self.threshold = 0.4
        self.labels = ["Unlikely", "Less likely","More likely"]

    def prepare_df_for_predict(self, temp_df):
        df = pd.DataFrame({
                       'text':temp_df["title"].values + "." +  temp_df["abstract"].values,
                        'title':temp_df["title"].values,
                        'abstract':temp_df["abstract"].values,
                        'label':[-1]*len(temp_df)})
        for column in ["text", "title", "abstract"]:
            df[column] = df[column].apply(text_normalizer.remove_html_tags)
        print(df['label'].value_counts())
        return df

    def prepare_df(self, filename):
        temp_df = excel_reader.ExcelReader().read_df_from_excel(filename)
        df = pd.DataFrame({
                       'text':temp_df["title"].values + "." +  temp_df["abstract"].values,
                        'title':temp_df["title"].values,
                        'abstract':temp_df["abstract"].values,
                        'label':temp_df["Label"].values})
        for column in ["text", "title", "abstract"]:
            df[column] = df[column].apply(text_normalizer.remove_html_tags)
        print(df['label'].value_counts())
        return df

    def evaluate_with_boost_and_threshold(self, test, use_tail = False):
        sg_boost_test_x =self.prepare_dataset_for_boosting(test, use_tail = use_tail)
        test_y = list(test[self.label_column].values)
        res = [1 if prob[1] >= self.threshold else 0 for prob in self.sgBoost.predict_proba(sg_boost_test_x)]
        print(confusion_matrix(test_y, res))
        print(classification_report(test_y, res))
        return (test_y, res)

    def predict_with_boost_and_threshold(self, test, use_tail = False):
        sg_boost_test_x =self.prepare_dataset_for_boosting(test, use_tail = use_tail)
        res_prob = self.sgBoost.predict_proba(sg_boost_test_x)
        res = [1 if prob[1] >= self.threshold else 0 for prob in res_prob]
        return res

    def predict_with_boost_and_threshold_with_degree(self, test, use_tail = False):
        sg_boost_test_x =self.prepare_dataset_for_boosting(test, use_tail = use_tail)
        res_prob = self.sgBoost.predict_proba(sg_boost_test_x)
        res = [self.labels[0] if prob[1] < 0.1 else (self.labels[2] if prob[1] >= self.threshold else self.labels[1]) for prob in res_prob]
        return res, res_prob

    def get_text_a(self, x):
       return x["title"]

    def get_text_b(self, x):
       return x["abstract"]

    def get_text_a_from_tuple(self, x):
        return x[1]

    def get_text_b_from_tuple(self, x):
        return x[2]

    def label_df_with_probabilities_to_use_full_text(self, articles_df):
        df = self.prepare_df_for_predict(articles_df)
        res_label, res_prob = self.predict_with_boost_and_threshold_with_degree(df, use_tail = True)

        articles_df["should_be_seen_in_full_text"] = res_label
        articles_df["should_be_seen_in_full_text_probability"] = [prob[1] for prob in res_prob]
        return articles_df
