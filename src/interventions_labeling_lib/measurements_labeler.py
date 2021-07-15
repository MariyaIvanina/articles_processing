from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

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
from bert_models.base_bert_model import BaseBertModel

#nlp = spacy.load('en_core_web_sm')
#lmtzr = WordNetLemmatizer()

class MeasurementsLabeler(BaseBertModel):

    def __init__(self, output_dir = "", label_list=[0,1], gpu_device_num = 1, batch_size = 16, max_seq_length = 128,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "", label_column = "Label"):
        print("Started BERT")
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,\
        bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column)

    def get_train_and_test_data_splitted(self, temp_df, all_dataset = False, test_size=0.05):
        if all_dataset:
            train_data = temp_df
            test_data = temp_df
        else:
            train_data_x, test_data_x,train_y,test_y = train_test_split([data[:-1] for data in temp_df.values], [data[-1] for data in temp_df.values], test_size=test_size)
            train_data = np.concatenate((train_data_x,[[val]for val in train_y]),axis=1)
            test_data = np.concatenate((test_data_x,[[val] for val in test_y]),axis=1)
        return pd.DataFrame(train_data, columns=temp_df.columns), pd.DataFrame(test_data, columns=temp_df.columns)

    def prepare_datasets(self, df_filename, all_dataset = False, test_size=0.05):
        temp_df = pd.read_excel(df_filename)
        temp_df = shuffle(temp_df)
        for i in range(len(temp_df)):
            temp_df["Number"].values[i] = str(temp_df["Number"].values[i])
        return self.get_train_and_test_data_splitted(temp_df, all_dataset, test_size)

    def get_text_a(self, x):
       return x["Sentence"]

    def get_text_b(self, x):
       return str(x["Number"]) +" " + x["Word Expression"]

    def get_text_a_from_tuple(self, x):
        return x[0]

    def get_text_b_from_tuple(self, x):
        return str(x[1]) + " " + str(x[2])

    def find_positions_of_words(self, text_words, marker_words_set):
        positions_found = {}
        for i in range(len(text_words)):
            for offset in range(5):
                text_to_check = " ".join(text_words[i:i+offset])
                if text_to_check in marker_words_set:
                    positions_found[text_to_check] = i
        for word in marker_words_set:
            if word not in positions_found:
                positions_found[word] = 0
        return positions_found

    def apply_only_closest_word_for_measurement(self, sent, num, measurements):
        res = []
        text_words = text_normalizer.get_stemmed_words_inverted_index(sent, r"\d+[.,]?\d+\)?%?|\w+")
        words_to_check = [measure[0] for measure in measurements if measure[1] == 1] + [num]
        positions_found = self.find_positions_of_words(text_words, words_to_check)
        num_pos = positions_found[num]
        min_dist, word_min = 1000, ""
        for word in positions_found:
            if word != num:
                if abs(positions_found[word] - num_pos) < min_dist:
                    min_dist, word_min = abs(positions_found[word] - num_pos), word
        for measure in measurements:
            res.append((measure[0], int(measure[0] == word_min), measure[2]))
        return res

    def identify_measurements(self, pairs_to_check):
        dict_measure = {}
        for idx, pair in enumerate(pairs_to_check):
            if pair[0] not in dict_measure:
                dict_measure[pair[0]] = {}
            if pair[1] not in dict_measure[pair[0]]:
                dict_measure[pair[0]][pair[1]] = []
            dict_measure[pair[0]][pair[1]].append((pair[2],pair[3],idx))
        for sent in dict_measure:
            for num in dict_measure[sent]:
                if len(dict_measure[sent][num]) > 1:
                    measures = self.apply_only_closest_word_for_measurement(sent, num, dict_measure[sent][num])
                    dict_measure[sent][num] = measures
        new_dict = {}
        for sent in dict_measure:
            for num in dict_measure[sent]:
                for res in dict_measure[sent][num]:
                    new_dict[res[-1]] = (sent, num, res[0],res[1])
        measurements = []
        for i in range(max(new_dict) +1):
            measurements.append(new_dict[i])
        return measurements

    def find_last_processed_id(self, folder_to_save):
        if len(os.listdir(folder_to_save)) == 0:
            return -1
        max_file_id = max([int(filename.replace(".pickle","")) for filename in os.listdir(folder_to_save)])
        return max_file_id

    def prepare_normalized_sentence_for_interventions_check(self, sentence):
        norm_sentence = text_normalizer.normalize_text(sentence)
        norm_sentence = " ".join(text_normalizer.get_stemmed_words_inverted_index(norm_sentence))
        return norm_sentence

    def prepare_sentences_by_tokenization(self, title, abstract):
        sentences = [title]
        sentences.extend(nltk.sent_tokenize(abstract))
        res = []
        for sentence in sentences:
            norm_sentence = self.prepare_normalized_sentence_for_interventions_check(sentence)
            for m in re.finditer(r"\b\d+[\w.,]*\d*\b", sentence):
                number = m.group(0)
                res.append((number, norm_sentence, sentence))
        return res

    def prepare_sentences_by_words_number_limit(self, title, abstract, words_limit):
        res = []
        norm_title = text_normalizer.normalize_text(title)
        norm_title = " ".join(text_normalizer.get_stemmed_words_inverted_index(norm_title))
        for m in re.finditer(r"\b\d+[\w.,]*\d*\b", title):
            number = m.group(0)
            res.append((number, norm_title, title))
        abstract_words_split = abstract.split()
        for i in range(len(abstract_words_split)):
            num = re.search(r"\b\d+[\w.,]*\d*\b", abstract_words_split[i])
            if num:
                start_ind = max(i-words_limit, 0)
                part_text = " ".join(abstract_words_split[start_ind:i+words_limit])
                part_text_normalized = self.prepare_normalized_sentence_for_interventions_check(part_text)
                res.append((num.group(0), part_text_normalized, part_text))
        return res

    def find_measurements_for_articles(self, articles_df, output_dir, column_to_check, 
            num_words_limit = None, num_words_limit_with_ids = None):
        folder_to_save = os.path.join(output_dir, column_to_check)
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        start_id = self.find_last_processed_id(folder_to_save) + 1
        cnt_art_res = {}
        cnrt_art_all = {}
        all_data = []
        sentId2docId = {}
        sentId = 0
        pairsToProcess = []
        time_start = time()
        for docId in range(start_id, len(articles_df)):
            limit_for_doc = None if num_words_limit is None else num_words_limit
            if num_words_limit_with_ids is not None:
                if doc in num_words_limit_with_ids:
                    limit_for_doc = num_words_limit_with_ids[doc]
            title = text_normalizer.remove_accented_chars(articles_df["title"].values[docId].lower()\
                if articles_df["title"].values[docId].isupper() else articles_df["title"].values[docId])
            abstract = text_normalizer.remove_accented_chars(articles_df["abstract"].values[docId])
            sentences_to_check = self.prepare_sentences_by_tokenization(title,\
                abstract) if limit_for_doc is None else\
            self.prepare_sentences_by_words_number_limit(title,\
                abstract, limit_for_doc)
            for number, norm_sentence, sentence in sentences_to_check:
                for key_word in articles_df[column_to_check].values[docId]:
                    interv_set = set([key_word])#.union(search_engine_inverted_index.extend_query_with_abbreviations(key_word, extend_with_abbreviations = True))
                    for interv in interv_set:
                        if interv in norm_sentence:
                            sentId2docId[sentId] = docId
                            sentId += 1
                            pairsToProcess.append((sentence,number,interv))

            cnrt_art_all[docId] = articles_df[column_to_check].values[docId]
            if docId % 500 == 0 or docId == len(articles_df) -1:
                print("Processed %d articles" % docId)
                print("Processing ", time()-time_start)
                time_start = time()
            if (docId % 500 == 0 or docId == len(articles_df) -1) and len(pairsToProcess) != 0:
                if len(pairsToProcess) % 2 != 0:
                    sentId2docId[sentId] = sentId2docId[sentId-1]
                    sentId += 1
                    pairsToProcess.append(pairsToProcess[-1])
                pred_vals = self.getPredictions(pairsToProcess)
                pairs_to_check = []
                for idx,pair in enumerate(pairsToProcess):
                    pairs_to_check.append(tuple(list(pair) + [pred_vals[idx], sentId2docId[idx]]))
                all_data.extend(pairs_to_check)
                print("Measurements were predicted")
                for i in range(len(pairs_to_check)):
                    if pairs_to_check[i][-1]:
                        if sentId2docId[i] not in cnt_art_res:
                            cnt_art_res[sentId2docId[i]] = set()
                        cnt_art_res[sentId2docId[i]].add(pairs_to_check[i][2])
                sentId2docId = {}
                sentId = 0
                pairsToProcess = []
                print("Predicting ", time()-time_start)
                time_start = time()
            if (docId % 3000 == 0 or docId == len(articles_df) -1):
                pickle.dump([cnt_art_res, cnrt_art_all, all_data],\
                            open(os.path.join(folder_to_save, "%d.pickle"%docId),"wb"))
                cnt_art_res = {}
                cnrt_art_all = {}
                all_data = []

    def label_measurement_columns(self, articles_df, folder,
            num_words_limit = None,
            num_words_limit_with_ids = None,
            mappings_for_measurements={"interventions_found_raw":"measurements_for_interventions",\
                                    "plant_products_search_details":"measurements_for_crops",\
                                    "animal_products_search_details":"measurements_for_crops"}):

        for column in mappings_for_measurements:
            print(column)
            self.find_measurements_for_articles(articles_df, folder, column,
                num_words_limit=num_words_limit, num_words_limit_with_ids=num_words_limit_with_ids)

        articles_df["measurements"] = ""
        for i in range(len(articles_df)):
            articles_df["measurements"].values[i] = set()

        
        for column_name in os.listdir(folder):
            assert column_name in mappings_for_measurements
            mapping_column_name = mappings_for_measurements[column_name] 
            articles_df[mapping_column_name] = ""
            for i in range(len(articles_df)):
                articles_df[mapping_column_name].values[i] = set()

        last_maxId = 0
        for column_name in os.listdir(folder):
            full_folder = os.path.join(folder, column_name)
            file_ids = sorted([int(filename.replace(".pickle","")) for filename in os.listdir(full_folder)])
            assert column_name in mappings_for_measurements
            mapping_column_name = mappings_for_measurements[column_name]
            for filename_id in file_ids:
                filename = str(filename_id) + ".pickle"
                cnt_art_res, cnrt_all, all_data = pickle.load(open(os.path.join(full_folder,filename),"rb"))
                max_docId = filename_id
                for i in range(last_maxId, max_docId+1):
                    if i in cnt_art_res:
                        res = set(articles_df[column_name].values[i]).intersection(cnt_art_res[i])
                        if len(res) > 0:
                            articles_df[mapping_column_name].values[i] = articles_df[mapping_column_name].values[i].union(res)
                last_maxId = max_docId
        for i in range(len(articles_df)):
            for column in set(mappings_for_measurements.values()):
                if len(articles_df[column].values[i]) > 0:
                    articles_df["measurements"].values[i].add(column[0].upper() + column.replace("_"," ")[1:])
                articles_df[column].values[i] = list(articles_df[column].values[i])
            articles_df["measurements"].values[i] = list(articles_df["measurements"].values[i])
        return articles_df

