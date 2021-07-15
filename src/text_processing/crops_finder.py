import pandas as pd
import re
import gensim
from text_processing import text_normalizer
import spacy
import os
from nltk.stem.wordnet import WordNetLemmatizer
from utilities import excel_writer
from utilities import excel_reader

nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

class CropsSearch:

    def __init__(self, search_engine_inverted_index, filename, threshold = 0.88, status_logger = None):
        self.search_engine_inverted_index = search_engine_inverted_index
        self.crops_names = {}
        self.threshold = threshold
        self.filename = filename
        self.initialize_crops_map()
        self.status_logger = status_logger

    def log_percents(self, percent):
        if self.status_logger is not None:
            self.status_logger.update_step_percent(percent)

    def initialize_crops_map(self):
        map_df = excel_reader.ExcelReader().read_df_from_excel(self.filename)
        self.real_names = {"level_3_term": {}, "narrow_name": {}, "broad_name": {}}

        for i in range(len(map_df)):
            level_3_term = map_df["level_3_term"].values[i].lower()
            if level_3_term not in self.crops_names:
                self.crops_names[level_3_term] = {}
            group_dict = self.crops_names[level_3_term]
            
            narrow_name = map_df["narrow_name"].values[i].lower()
            if narrow_name not in group_dict:
                group_dict[narrow_name] = []
            group_dict[narrow_name].append(map_df["broad_name"].values[i].lower())
            self.real_names["narrow_name"][narrow_name] = map_df["narrow_name"].values[i]
            self.real_names["broad_name"][map_df["broad_name"].values[i].lower()] = map_df["broad_name"].values[i]
            self.real_names["level_3_term"][level_3_term] = map_df["level_3_term"].values[i]


    def map_level_3_terms(self, articles_df, articles_with_crops_level_3, found_crops, column_name, keep_hierarchy=False):
        column_name_details = column_name + "_details"
        column_name_broad = column_name + "_broad"
        for level_3_term in self.crops_names:
            articles_with_keyword = self.search_engine_inverted_index.find_articles_with_keywords(
                [level_3_term], self.threshold, extend_with_abbreviations=False)
            for article_index in articles_with_keyword:
                if articles_df[column_name].values[article_index] == "":
                    articles_df[column_name].values[article_index] = set()
                if articles_df[column_name_details].values[article_index] == "":
                    articles_df[column_name_details].values[article_index] = set()
                articles_df[column_name_details].values[article_index].add(level_3_term)
                if keep_hierarchy:
                    if articles_df[column_name_broad].values[article_index] == "":
                        articles_df[column_name_broad].values[article_index] = set()
                    articles_df[column_name_broad].values[article_index].add(self.real_names["level_3_term"][level_3_term])
                    articles_df[column_name].values[article_index].add(self.real_names["level_3_term"][level_3_term] + "/")
                else:
                    articles_df[column_name].values[article_index].add(level_3_term)
                if level_3_term not in found_crops:
                    found_crops[level_3_term] = set()
                found_crops[level_3_term].add(article_index)
        return articles_df, found_crops

    def find_crops(self, articles_df, column_name, print_all = False, keep_hierarchy=False):
        found_crops = {}
        column_name_details = column_name + "_details"
        column_name_broad = column_name + "_broad"
        articles_df[column_name] = ""
        articles_df[column_name_details] = ""
        if keep_hierarchy:
            articles_df[column_name_broad] = ""
        articles_with_crops_level_3 = {}
        for idx, level_3_term in enumerate(self.crops_names):
            if idx % 10 == 0:
                self.log_percents(idx / len(self.crops_names)*90)
            for narrow_key, broad_values in self.crops_names[level_3_term].items():
                for key, values in [(narrow_key, broad_values)] + [(broad_name, [broad_name])for broad_name in broad_values]:
                    for article_index in self.search_engine_inverted_index.find_articles_with_keywords(
                            [key], self.threshold, extend_with_abbreviations=False):
                        if articles_df[column_name].values[article_index] == "":
                            articles_df[column_name].values[article_index] = set()
                        if articles_df[column_name_details].values[article_index] == "":
                            articles_df[column_name_details].values[article_index] = set()
                        if key not in found_crops:
                            found_crops[key] = set()
                        found_crops[key].add(article_index)
                        if level_3_term not in articles_with_crops_level_3:
                            articles_with_crops_level_3[level_3_term] = set()
                        articles_with_crops_level_3[level_3_term].add(article_index)
                        articles_df[column_name_details].values[article_index].add(key)
                        if keep_hierarchy:
                            if articles_df[column_name_broad].values[article_index] == "":
                                articles_df[column_name_broad].values[article_index] = set()
                            articles_df[column_name_broad].values[article_index].add(self.real_names["level_3_term"][level_3_term])
                            articles_df[column_name].values[article_index].add(self.real_names["level_3_term"][level_3_term] + "/")
                            for broad_name in values:
                                articles_df[column_name].values[article_index].add(
                                    self.real_names["level_3_term"][level_3_term] + "/" + self.real_names["broad_name"][broad_name])
                        else:
                            for broad_name in values:
                                articles_df[column_name].values[article_index].add(broad_name)

        articles_df, found_crops = self.map_level_3_terms(
            articles_df, articles_with_crops_level_3, found_crops, column_name, keep_hierarchy=keep_hierarchy)
        total = 0
        for i in range(len(articles_df)):
            for column in [column_name, column_name_details, column_name_broad]:
                if column not in articles_df.columns:
                    continue
                if articles_df[column].values[i] != "":
                    articles_df[column].values[i] = list(articles_df[column].values[i])
                    if column == column_name:
                        total += 1
                else:
                    articles_df[column].values[i] = []
        print("Found crops: " + column_name)
        if print_all:
            for key in found_crops:
                print(key, len(found_crops[key]))
        print("Number of articles that have %s : %d" %(column_name, total))
        return articles_df