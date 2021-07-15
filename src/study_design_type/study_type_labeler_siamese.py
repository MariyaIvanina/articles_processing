from bert_models.base_bert_model_with_boosting import BaseBertModelWithBoost
from bert_models.base_bert_model_siamese import BaseBertModel

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

class StudyTypeLabeler(BaseBertModel):

    def __init__(self, output_dir, label_list = list(StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER.values()),
            gpu_device_num_hub = 0, gpu_device_num = 1, batch_size = 16, max_seq_length = 256,
            bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "../scibert_scivocab_uncased",
            label_column = "label", use_concat_results = False):
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num_hub=gpu_device_num_hub, 
            gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,
            bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column,
            use_concat_results=use_concat_results)

    def get_df_from_file(self, filename, without_labels = False):
        temp_df = excel_reader.ExcelReader().read_df_from_excel(filename)
        if not without_labels:
            temp_df = shuffle(temp_df)
        return self.get_prepared_df(temp_df, without_labels)


    def get_prepared_df(self, temp_df, without_labels = False):
        df = pd.DataFrame({
                        'title':temp_df["title"].values,
                        'abstract':temp_df["abstract"].values,
                        'text':temp_df["title"].values + "." +  temp_df["abstract"].values,
                        'label':[(StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER[val] if val in StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER else -1) for val in temp_df["extended_study_type"].values] if not without_labels else [-1]*len(temp_df)})
        for column in ["text", "title", "abstract"]:
            df[column] = df[column].apply(text_normalizer.remove_html_tags)
        print(df['label'].value_counts())
        return df

    def get_text_a(self, x):
       return x["title"]

    def get_text_b(self, x):
       return x["abstract"]

    def get_text_a_from_tuple(self, x):
        return x[0]

    def get_text_b_from_tuple(self, x):
        return x[1]

    def save_file_with_predictions(self, df, filename, withBoost = False, use_tail = False):
        res_prob, res_label, res_y = self.predict_for_df(df) if not withBoost else self.predict_with_boosting(df, with_head_tail = use_tail)

        pred_test_vals = []
        for i, test_val in enumerate(res_y):
            pred_test_vals.append([df["title"].values[i], df["abstract"].values[i]] + list(res_prob[i]) + [StudyTypeLabels.STUDY_TYPE_NUMBER_TO_LABEL[res_label[i]]])
        excel_writer.ExcelWriter().save_data_in_excel(pred_test_vals,["title", "abstract"]+list(StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER.keys())+["Label"], filename,\
         column_probabilities=StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER.keys(), column_interventions=["Label"], dict_with_colors= StudyTypeLabels.STUDY_TYPE_LABELS_COLOR)

    def save_file_with_predictions_with_label(self, df, filename, withBoost = False, use_tail = False):
        res_prob, res_label, res_y = self.predict_for_df(df) if not withBoost else  self.predict_with_boosting(df, with_head_tail = use_tail)
        
        pred_test_vals = []
        for i in range(len(df)):
            pred_test_vals.append([df["title"].values[i], df["abstract"].values[i]] + list(res_prob[i]) + [StudyTypeLabels.STUDY_TYPE_NUMBER_TO_LABEL[res_label[i]], df.values[i][-1].strip() if type(df.values[i][-1]) == str else StudyTypeLabels.STUDY_TYPE_NUMBER_TO_LABEL[df.values[i][-1]]])
        excel_writer.ExcelWriter().save_data_in_excel(pred_test_vals,["title", "abstract"]+list(StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER.keys())+["Label", "Original label"], filename, \
            column_probabilities=StudyTypeLabels.STUDY_TYPE_LABEL_TO_NUMBER.keys(), column_interventions=["Label","Original label"], dict_with_colors= StudyTypeLabels.STUDY_TYPE_LABELS_COLOR) 

    def generate_test_file(self, test_filename, new_test_filename, size = 50, withBoost = False, use_tail = False):
        test_df = excel_reader.ExcelReader().read_df_from_excel(test_filename)
        test_df = shuffle(test_df)
        part_test_df = self.get_prepared_df(test_df)
        part_test_df = part_test_df[part_test_df[self.label_column] != -1]
        self.save_file_with_predictions(part_test_df[:size], new_test_filename, withBoost = withBoost, use_tail = use_tail)
        title_dict = set()
        for i in range(size):
            title_dict.add(part_test_df["title"].values[i])
        indices_to_keep = []
        for i in range(len(test_df)):
            if test_df["title"].values[i] not in title_dict:
                indices_to_keep.append(i)
        test_df_1 = test_df.take(indices_to_keep)
        excel_writer.ExcelWriter().save_df_in_excel(test_df_1, test_filename)