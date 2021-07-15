import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from outcomes_modelling import outcomes_extracted_labeler
from outcomes_modelling import outcomes_extracted_labeler_plus_keywords
from outcomes_modelling import outcomes_sentence_labeler
from outcomes_modelling import outcomes_sentence_labeler_plus_NER
from sklearn.metrics import f1_score
import shutil
import nltk

def highlight_max(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        print(data.max())
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

class OutcomesFullLogicLabeller:

    def __init__(self, model_folder, ner_model_folder, threshold=0.5):
        self.model_folder = model_folder
        self.ner_model_folder = ner_model_folder
        self.labels = ['Livelihood', 'Soil health', 'Fertilizer use', 'Water use', 'Gender',
           'Greenhouse gas emissions', 'Market access', 'Nutrition',
           'Production', 'Resilience', 'Knowledge sharing', 'Practice change',
           'Social inclusion', 'Poverty reduction', 'Environment impact']
        self.best_phrase_classifications = {}
        self.chosen_models_info = {}
        self.threshold = threshold
        self.load_model()

    def train_model(self, train_data_file):
        train_data_full = pickle.load(open(train_data_file, "rb"))
        TRAIN_DATA, outcomes_train_label_data, outcomes_sentences_train_data = train_data_full[:3]
        EVAL_DATA, outcomes_eval_label_data, outcomes_sentences_eval_data = train_data_full[3:]
        TRAIN_DATA = shuffle(TRAIN_DATA)
        outcomes_train_label_data = shuffle(outcomes_train_label_data)
        outcomes_sentences_train_data = shuffle(outcomes_sentences_train_data)
        print("Train data distribution")
        print(pd.DataFrame(outcomes_train_label_data, columns=["Text", "Label"])["Label"].value_counts())
        print("Valid data distribution")
        print(pd.DataFrame(outcomes_eval_label_data, columns=["Text", "Label"])["Label"].value_counts())

        results = []
        model_names = []
        result_model_info = {}
        for label_to_train in self.labels:
            print("###")
            print(label_to_train)
            result_model_info[label_to_train] = {}

            model_names = []

            name = "Only extracted texts"
            print(name)
            one_row = [label_to_train]
            _outcomes_extracted_labeler = outcomes_extracted_labeler.OutcomesExtractedLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_1"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_extracted_labeler.prepare_data_df(outcomes_train_label_data, label_to_train)
            print(train_data_df["label"].value_counts())
            test_data_df =  _outcomes_extracted_labeler.prepare_data_df(outcomes_eval_label_data, label_to_train)
            res = _outcomes_extracted_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"extracted words classification": "outcomes_12.12_%s_1"%label_to_train}
            
            name = "Only extracted texts + proportion 0.1"
            print(name)
            _outcomes_extracted_labeler = outcomes_extracted_labeler.OutcomesExtractedLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_1_1"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_extracted_labeler.prepare_data_df(outcomes_train_label_data, label_to_train,
                                                                        proportion_to_check=0.1)
            print(train_data_df["label"].value_counts())
            test_data_df =  _outcomes_extracted_labeler.prepare_data_df(outcomes_eval_label_data, label_to_train)
            res = _outcomes_extracted_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"extracted words classification": "outcomes_12.12_%s_1_1"%label_to_train}
            
            name = "Only extracted texts + keywords additional"
            print(name)
            _outcomes_extracted_labeler = outcomes_extracted_labeler_plus_keywords.OutcomesExtractedLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_2"%label_to_train), gpu_device_num=0,
                path_to_keywords="../data/outcomes_dictionary.xlsx")
            train_data_df = _outcomes_extracted_labeler.prepare_data_df(outcomes_train_label_data, label_to_train)
            print(train_data_df["label"].value_counts())
            test_data_df =  _outcomes_extracted_labeler.prepare_data_df(outcomes_eval_label_data, label_to_train)
            res = _outcomes_extracted_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"extracted words classification": "outcomes_12.12_%s_2"%label_to_train}
            
            name = "Only extracted texts + keywords additional + proportion 0.1"
            print(name)
            _outcomes_extracted_labeler = outcomes_extracted_labeler_plus_keywords.OutcomesExtractedLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_2_1"%label_to_train), gpu_device_num=0,
                path_to_keywords="../data/outcomes_dictionary.xlsx")
            train_data_df = _outcomes_extracted_labeler.prepare_data_df(outcomes_train_label_data, label_to_train,
                                                                        proportion_to_check=0.1)
            print(train_data_df["label"].value_counts())
            test_data_df =  _outcomes_extracted_labeler.prepare_data_df(outcomes_eval_label_data, label_to_train)
            res = _outcomes_extracted_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"extracted words classification": "outcomes_12.12_%s_2_1"%label_to_train}
            
            name = "BERT(trained on only sentences with outcomes)"
            print(name)
            _outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_3"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_train_data, label_to_train,
                                                                                 include_empty_sentences=False)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"sentences classification": "outcomes_12.12_%s_3"%label_to_train}
            one_row.append(f1_score(res[2], res[1], average="macro"))
            model_names.append(name)
            
            name = "BERT(trained on only sentences with outcomes) + proportion 0.1"
            print(name)
            _outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_3_1"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_train_data, label_to_train,
                                                                                 include_empty_sentences=False, proportion_to_check=0.1)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"sentences classification": "outcomes_12.12_%s_3_1"%label_to_train}
            one_row.append(f1_score(res[2], res[1], average="macro"))
            model_names.append(name)
            
            name = "BERT(trained on all sentences)"
            print(name)
            _outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_4"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_train_data,
                                                                                 label_to_train, include_empty_sentences=True)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"sentences classification": "outcomes_12.12_%s_4"%label_to_train}
            one_row.append(f1_score(res[2], res[1], average="macro"))
            model_names.append(name)
            
            name = "BERT(trained on all sentences) + proportion 0.1"
            print(name)
            _outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
                os.path.join(self.model_folder, "outcomes_12.12_%s_4_1"%label_to_train), gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_train_data, label_to_train,
                                                                                 include_empty_sentences=True, proportion_to_check=0.1)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler.prepare_df_data_sentences(outcomes_sentences_eval_data, label_to_train,
                                                                                include_empty_sentences=True)
            res = _outcomes_sentence_labeler.train_model(train_data_df, test_data_df)
            result_model_info[label_to_train][name] = {"sentences classification": "outcomes_12.12_%s_4_1"%label_to_train}
            one_row.append(f1_score(res[2], res[1], average="macro"))
            model_names.append(name)
            
            name = "NER + BERT classification"
            print(name)
            _outcomes_sentence_labeler_plus_NER = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder, "outcomes_12.12_%s_1"%label_to_train),
                path_to_NER=self.ner_model_folder, gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_train_data, label_to_train, include_empty_sentences=True)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler_plus_NER.evaluate_model(test_data_df)
            result_model_info[label_to_train][name] = {
                "extracted words classification": "outcomes_12.12_%s_1"%label_to_train,
                 "NER": self.ner_model_folder}
            one_row.append(f1_score(list(test_data_df["label"].values), res, average="macro"))
            model_names.append(name)
            
            name = "NER + BERT classification + proportion 0.1"
            _outcomes_sentence_labeler_plus_NER = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder, "outcomes_12.12_%s_1_1"%label_to_train),
                path_to_NER=self.ner_model_folder, gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_train_data, label_to_train, include_empty_sentences=True)
            test_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler_plus_NER.evaluate_model(test_data_df)
            result_model_info[label_to_train][name] = {
                "extracted words classification": "outcomes_12.12_%s_1_1"%label_to_train,
                 "NER": self.ner_model_folder}
            one_row.append(f1_score(list(test_data_df["label"].values), res, average="macro"))
            model_names.append(name)
            
            name = "NER + BERT classification + keywords"
            _outcomes_sentence_labeler_plus_NER = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder, "outcomes_12.12_%s_2"%label_to_train),
                path_to_NER=self.ner_model_folder, gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_train_data, label_to_train, include_empty_sentences=True)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler_plus_NER.evaluate_model(test_data_df)
            result_model_info[label_to_train][name] = {
                "extracted words classification": "outcomes_12.12_%s_2"%label_to_train,
                 "NER": self.ner_model_folder}
            one_row.append(f1_score(list(test_data_df["label"].values), res, average="macro"))
            model_names.append(name)
            
            name = "NER + BERT classification + keywords + proportion 0.1"
            print(name)
            _outcomes_sentence_labeler_plus_NER = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder, "outcomes_12.12_%s_2_1"%label_to_train),
                path_to_NER=self.ner_model_folder, gpu_device_num=0)
            train_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_train_data, label_to_train, include_empty_sentences=True)
            print(train_data_df["label"].value_counts())
            test_data_df = _outcomes_sentence_labeler_plus_NER.prepare_df_data_sentences(
                outcomes_sentences_eval_data, label_to_train, include_empty_sentences=True)
            res = _outcomes_sentence_labeler_plus_NER.evaluate_model(test_data_df)
            result_model_info[label_to_train][name] = {
                "extracted words classification": "outcomes_12.12_%s_2_1"%label_to_train,
                 "NER": self.ner_model_folder}
            one_row.append(f1_score(list(test_data_df["label"].values), res, average="macro"))
            model_names.append(name)
            
            results.append(one_row)

        results_df = pd.DataFrame(results, columns=["Label"] + model_names)
        results_df = results_df.set_index("Label")
        results_df.loc['Average F1']= results_df.mean(numeric_only=True, axis=0)
        chosen_models_ind = np.argmax(results_df.values, axis=1)
        pickle.dump([results_df,
            pd.DataFrame(outcomes_train_label_data, columns=["Text", "Label"])["Label"].value_counts(),
            pd.DataFrame(outcomes_eval_label_data, columns=["Text", "Label"])["Label"].value_counts()], open(os.path.join(
                self.model_folder, "metrics_results.pickle"), "wb"))

        
        print("Chosen models")
        for idx, label in enumerate(results_df.index.get_values()):
            if label not in self.labels:
                continue
            model_name = model_names[chosen_models_ind[idx]]
            print(label, " ## ", model_name)
            self.chosen_models_info[label] = {"name": model_name, "settings": result_model_info[label][model_name]}

        print("Chosen phrase models")
        part_result_df = results_df[[column for column in model_names if "NER" in column]].copy()
        part_chosen_models_ind = np.argmax(part_result_df.values, axis=1)
        for idx, label in enumerate(part_result_df.index.get_values()):
            if label not in self.labels:
                continue
            model_name = part_result_df.columns[part_chosen_models_ind[idx]]
            print(label, " ## ", model_name)
            self.best_phrase_classifications[label] = {"name": model_name, "settings": result_model_info[label][model_name]}

        self.remove_unnecessary_folders()
        return self.print_statistics()

    def print_statistics(self):
        all_data = pickle.load(open(os.path.join(self.model_folder, "metrics_results.pickle"), "rb"))
        results_df = all_data[0]
        print("Mean the best F1 score: ", np.mean(results_df.max(numeric_only=True, axis=1)))
        return results_df.style.apply(highlight_max, axis=1)

    def save_model(self):
        pickle.dump(self.chosen_models_info, open(os.path.join(self.model_folder, "chosen_model_info.pickle"), "wb"))
        pickle.dump(self.best_phrase_classifications, open(os.path.join(self.model_folder, "best_phrase_classifications.pickle"), "wb"))

    def load_model(self):
        file_to_load = os.path.join(self.model_folder, "chosen_model_info.pickle")
        self.chosen_models_info = {}
        if os.path.exists(file_to_load):
            self.chosen_models_info = pickle.load(open(file_to_load, "rb"))
        file_to_load = os.path.join(self.model_folder, "best_phrase_classifications.pickle")
        self.best_phrase_classifications = {}
        if os.path.exists(file_to_load):
            self.best_phrase_classifications = pickle.load(open(file_to_load, "rb"))

    def remove_unnecessary_folders(self):
        all_folders = set()
        for label in self.labels:
            for _dict in [self.chosen_models_info, self.best_phrase_classifications]:
                    for name, path in _dict[label]["settings"].items():
                        all_folders.add(path)
        if len(all_folders) > 0:
            for folder in os.listdir(self.model_folder):
                if os.path.isdir(os.path.join(self.model_folder, folder)) and folder not in all_folders:
                    #shutil.rmtree(os.path.join(self.model_folder, folder), ignore_errors=True)
                    print("Deleted ", folder)

    def predict(self, sentences, label):
        assert label in self.chosen_models_info
        data_df = pd.DataFrame([(sent, -1) for sent in sentences], columns=["text", "label"])
        outcomes_details = []
        if len(self.chosen_models_info[label]["settings"]) == 1:
            _outcomes_sentence_labeler = outcomes_sentence_labeler.OutcomesSentenceLabeler(
                os.path.join(self.model_folder,
                    self.chosen_models_info[label]["settings"]["sentences classification"]), gpu_device_num=0)
            res_prob, res_label, res_y = _outcomes_sentence_labeler.predict_for_df(data_df)
            for i in range(len(res_label)):
                outcomes_details.append(["[%s]%s"%(label, sentences[i])] if res_prob[i][1] >= self.threshold else [])
            res_label = [label if r[1] >= self.threshold else "" for r in res_prob]
        if len(self.chosen_models_info[label]["settings"]) == 2:
            _outcomes_sentence_labeler = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder,
                    self.chosen_models_info[label]["settings"]["extracted words classification"]),
                path_to_NER=self.chosen_models_info[label]["settings"]["NER"], gpu_device_num=0,
                threshold=self.threshold)
            res_label = _outcomes_sentence_labeler.predict_for_sentences(data_df.values, only_res=False)
            outcomes_details = []
            for full_res, extracted_parts in res_label:
                outcomes_details.append(["[%s]%s"%(label, part) for part, _label in extracted_parts if _label])
            res_label = [label if r[0] else "" for r in res_label]
        return res_label, outcomes_details

    def predict_all_labels(self, texts, found_labels=None):
        sent2partsLabels = {}
        for idx in range(len(texts)):
            sent2partsLabels[idx] = (set(), set())
        for label in self.labels:
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
            res_label, outcomes_details = self.predict(texts_to_sentences, label)
            for idx in sent2parts:
                for sent_id in sent2parts[idx]:
                    if res_label[sent_id]:
                        sent2partsLabels[idx][0].add(res_label[sent_id])
                        sent2partsLabels[idx][1].update(outcomes_details[sent_id])
        return [list(sent2partsLabels[idx][0]) for idx in range(len(texts))], [list(sent2partsLabels[idx][1]) for idx in range(len(texts))]

    def predict_all_labels_old(self, texts, details=False, found_labels=None):
        sent2partsLabels = {}
        for idx in range(len(texts)):
            sent2partsLabels[idx] = set()
        for label in self.labels:
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
            outcomes_details = self.predict_details(
                texts_to_sentences, label)  if details else self.predict(texts_to_sentences, label)
            for idx in sent2parts:
                for sent_id in sent2parts[idx]:
                    if outcomes_details[sent_id]:
                        if type(outcomes_details[sent_id]) == str:
                            sent2partsLabels[idx].add(outcomes_details[sent_id])
                        else:
                            sent2partsLabels[idx].update(outcomes_details[sent_id])
        return [list(sent2partsLabels[idx]) for idx in range(len(texts))]

    def predict_details(self, sentences, label):
        assert label in self.best_phrase_classifications
        data_df = pd.DataFrame([(sent, -1) for sent in sentences], columns=["text", "label"])
        _outcomes_sentence_labeler = outcomes_sentence_labeler_plus_NER.OutcomesSentenceLabelerPlusNer(
                os.path.join(self.model_folder,
                    self.best_phrase_classifications[label]["settings"]["extracted words classification"]),
                path_to_NER=self.best_phrase_classifications[label]["settings"]["NER"], gpu_device_num=0)
        res_full_label = _outcomes_sentence_labeler.predict_for_sentences(data_df.values, only_res=False)
        outcomes_details = []
        for full_res, extracted_parts in res_full_label:
            outcomes_details.append(["[%s]%s"%(label, part) for part, _label in extracted_parts if _label])
        return outcomes_details

