import pandas as pd
import re
import gensim
from text_processing import text_normalizer
from text_processing import crops_finder
import textdistance
import spacy
import os
from nltk.stem.wordnet import WordNetLemmatizer
from utilities import excel_writer
from utilities import excel_reader
from datetime import datetime

nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

class CropsExtractor:

    def __init__(self, search_engine_inverted_index, filename, crop_type):
        self.search_engine_inverted_index = search_engine_inverted_index
        self.google_model = gensim.models.Word2Vec.load(os.path.join("../model/google_plus_our_dataset/", "google_plus_our_dataset.model"))
        self.crop_type = crop_type
        self.filename = filename
        crops_mapper = crops_finder.CropsSearch(search_engine_inverted_index, filename)
        crops_mapper.initialize_crops_map()
        self.crops_names = crops_mapper.crops_names

    def has_main_word(self, expression, word):
        for noun_chunk in nlp(expression).noun_chunks:
            if re.search(r"\b%se?s?\b"%word, noun_chunk.root.text) != None:
                return True
        return False

    def are_word_similar_by_meaning(self, first_word, second_word, threshold = 0.2):
        similarity = 0.0
        if first_word in self.google_model.wv and second_word in self.google_model.wv:
            similarity = self.google_model.wv.similarity(first_word, second_word)
        return similarity > 0.2

    def is_found_by_specific_pattern(self, sentences, pattern_numbers):
        for sentence in sentences:
            if sentence[2] in pattern_numbers:
                return True
        return False

    def has_abbreviations(self, sentence):
        for word in sentence.split():
            if text_normalizer.is_abbreviation(word):
                return True
        return False

    def find_hyponyms_for_word(self, hyponyms_search, word_to_check, specific_patterns = [0,1,2,3]):
        filter_words = ["colour","sesquiterpenes","diterpenes","triterpenoids","sterol","sausage","marbling"]
        word_to_check = text_normalizer.normalize_sentence(word_to_check)
        hyponym_words = set()
        for key in hyponyms_search.dict_hyponyms.keys():
            if word_to_check in key and self.has_main_word(key, word_to_check):
                for key_word in hyponyms_search.dict_hyponyms[key]:
                    if word_to_check not in key_word and self.is_found_by_specific_pattern(hyponyms_search.dict_hyponyms[key][key_word],specific_patterns) and \
                    self.are_word_similar_by_meaning(word_to_check, key_word) and len(key.split()) < 5 and not self.has_abbreviations(key_word) and\
                    len(self.search_engine_inverted_index.get_articles_by_word(key_word)) < min(0.1*self.search_engine_inverted_index.total_articles_number, 3000) \
                    and len(self.search_engine_inverted_index.get_articles_by_word(key_word)) < len(self.search_engine_inverted_index.get_articles_by_word(word_to_check)):
                        if key_word not in filter_words:
                            hyponym_words.add(key_word)
                            print(key, "---", key_word, "---", self.google_model.wv.similarity(word_to_check, key_word))
                        #print(hyponyms_search.dict_hyponyms[key][key_word])
        return hyponym_words

    def get_normalized_keys(self, dict_to_check):
        words_set = set()
        for key in dict_to_check:
            words_set.add(text_normalizer.normalize_sentence(key))
        return words_set

    def find_crops_from_hyponyms(self, hyponyms_search, dict_to_check):
        new_mappings_to_add = set()
        for word in self.find_hyponyms_for_word(hyponyms_search, "crop") - dict_to_check:
            exists = False
            for key in dict_to_check:
                if textdistance.levenshtein.normalized_similarity(word, key) >= 0.8:
                    exists = True
            if not exists:
                print(word, len(self.search_engine_inverted_index.find_articles_with_keywords([word],0.88,extend_with_abbreviations=False)))
                if self.google_model.wv.similarity(self.crop_type, word) > 0.4:
                    new_mappings_to_add.add((word,word,word))
        return new_mappings_to_add

    def find_crop_names_by_hierarchy(self, hyponyms_search):
        dict_to_check = set()
        for level_1_name in self.crops_names:
            dict_to_check.add(level_1_name)
            dict_to_check = dict_to_check.union(self.get_normalized_keys(self.crops_names[level_1_name]))
            for level_2_name in self.crops_names[level_1_name]:
                dict_to_check = dict_to_check.union(self.get_normalized_keys(self.crops_names[level_1_name][level_2_name]))

        new_mappings_to_add = set()
        for level_2_name in self.crops_names:
            print(level_2_name)
            if level_2_name in ["grain"]:
                continue
            hyponyms_for_word = self.find_hyponyms_for_word(hyponyms_search, level_2_name)

            if len(hyponyms_for_word - dict_to_check) > 0:
                print(hyponyms_for_word - dict_to_check)
                for word in (hyponyms_for_word - dict_to_check):
                    print("###","###",word, len(self.search_engine_inverted_index.find_articles_with_keywords([word],0.85, extend_with_abbreviations=False)))
                    mappings = None 
                    for key in self.crops_names[level_2_name]:
                        if (re.search(r"\b%se?s?\b"%word, key) is not None and not self.has_main_word(key, word)) or textdistance.levenshtein.normalized_similarity(
                                word, key) >= 0.8:
                            print("Mapped to ", self.crops_names[level_2_name][key])
                            mappings = self.crops_names[level_2_name][key]
                            break
                    dict_to_check.add(word)
                    if mappings is not None:
                        for mapping_name in mappings:
                            new_mappings_to_add.add((word, mapping_name, level_2_name))
                            print((word, mapping_name, level_2_name))
                    else:
                        new_mappings_to_add.add((word, word, level_2_name))
                        print((word, word, level_2_name))
        new_mappings_to_add = new_mappings_to_add.union(self.find_crops_from_hyponyms(hyponyms_search,dict_to_check))
        return new_mappings_to_add

    def save_mappings(self, new_mappings_to_add):
        new_file_name = "{}_{}.{}".format(".".join(self.filename.split(".")[:-1]), str(int(datetime.timestamp(datetime.now()))),self.filename.split(".")[-1])
        map_df = excel_reader.ExcelReader().read_df_from_excel(self.filename)
        map_df = pd.DataFrame(new_mappings_to_add, columns=list(map_df.columns))
        excelWriter = excel_writer.ExcelWriter()
        excelWriter.save_df_in_excel(map_df, new_file_name)