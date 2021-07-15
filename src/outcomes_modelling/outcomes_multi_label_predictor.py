import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from outcomes_modelling import outcomes_sentence_labeler
from sklearn.metrics import f1_score
import shutil
import nltk

class OutcomesMultiLabelPredictor:

    def __init__(self, model_folder, ner_model_folder="", threshold=0.5):
        self.model_folder = model_folder
        self.ner_model_folder = ner_model_folder
        self.labels = ['Livelihood', 'Soil health', 'Fertilizer use', 'Water use', 'Gender',
           'Greenhouse gas emissions', 'Market access', 'Nutrition',
           'Production', 'Resilience', 'Knowledge sharing', 'Practice change',
           'Social inclusion', 'Poverty reduction', 'Environment impact']
        self.threshold = threshold
        self.outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
            model_folder,
            gpu_device_num=0, multilabel=True, label_list=list(range(len(self.labels))),
            epoch_num=15)

    def predict(self, sentences):
        data_df = pd.DataFrame([(sent, -1) for sent in sentences], columns=["text", "label"])
        res_prob, res_label, _ = self.outcomes_sentence_labeler.predict_for_df(data_df)
        outcomes_details = []
        outcomes_found = []
        for i in range(len(res_label)):
            outcomes_label_per_sentence = []
            outcomes_details_per_sentence = []
            for idx in range(len(self.labels)):
                if res_prob[i][idx] >= self.threshold:
                    outcomes_details_per_sentence.append("[%s]%s"%(self.labels[idx], sentences[i]))
                    outcomes_label_per_sentence.append(self.labels[idx])
            outcomes_details.append(outcomes_details_per_sentence)
            outcomes_found.append(outcomes_label_per_sentence)
        return outcomes_found, outcomes_details

    def predict_all_labels(self, texts, found_labels=None):
        sent2partsLabels = {}
        for idx in range(len(texts)):
            sent2partsLabels[idx] = (set(), set())
        texts_to_sentences = []
        id_ = 0
        sent2parts = {}
        for idx, text in enumerate(texts):
            sent2parts[idx] = []
            if found_labels is None or label in found_labels[idx]:
                for sent in nltk.sent_tokenize(text):
                    sent2parts[idx].append(id_)
                    texts_to_sentences.append(sent)
                    id_ += 1
        res_label, outcomes_details = self.predict(texts_to_sentences)
        for idx in sent2parts:
            for sent_id in sent2parts[idx]:
                if res_label[sent_id]:
                    sent2partsLabels[idx][0].update(res_label[sent_id])
                    sent2partsLabels[idx][1].update(outcomes_details[sent_id])
        return [list(sent2partsLabels[idx][0]) for idx in range(len(texts))], [list(sent2partsLabels[idx][1]) for idx in range(len(texts))]

