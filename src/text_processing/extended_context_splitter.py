import sys
sys.path.append('../src')

from text_processing import text_normalizer
from utilities import excel_writer, excel_reader
import argparse
import pandas as pd
import pickle
import os
from interventions_labeling_lib import hyponym_statistics
from time import time
import nltk
import re

class ExtendedContextSplitter:

    def __init__(self):
        stopwords_crops = set(excel_reader.ExcelReader().read_df_from_excel("../data/Filter_Geo_Names.xlsx")["Word"].values)
        for file in ["map_animals.xlsx", "map_plant_products_wo_foreign.xlsx", "map_animal_products_wo_foreign.xlsx"]:
            df_1 = excel_reader.ExcelReader().read_df_from_excel(os.path.join("../data", file))
            stopwords_crops = stopwords_crops.union(set(df_1["narrow_name"].values))
            stopwords_crops = stopwords_crops.union(set(df_1["broad_name"].values))
            stopwords_crops = stopwords_crops.union(set(df_1["level_3_term"].values))

        self.stopwords_crops_normalized = set()
        for word in stopwords_crops:
            self.stopwords_crops_normalized.add(text_normalizer.normalize_sentence(word))
        self.stopwords_crops_normalized = self.stopwords_crops_normalized.union(
            set(["aquaculture product", "including"])) - set(["fishery", "fishing", "cropping", "seed"])

        self.all_crops = set()
        for file in ["map_animals.xlsx", "map_plant_products_wo_foreign.xlsx", "map_animal_products_wo_foreign.xlsx"]:
            df_1 = excel_reader.ExcelReader().read_df_from_excel(os.path.join("../data", file))
            self.all_crops = self.all_crops.union(set(df_1["narrow_name"].values))
            self.all_crops = self.all_crops.union(set(df_1["broad_name"].values))
            self.all_crops = self.all_crops.union(set(df_1["level_3_term"].values))
        self.all_crops = set([text_normalizer.normalize_sentence(crop) for crop in self.all_crops]) - set([""])
        self._hyp_stat = hyponym_statistics.HyponymStatistics({}, {}, {}, {}, {})

    def clean_from_filter_words(self, sentence, filter_words):
        words = text_normalizer.normalize_sentence(sentence).split()
        filtered_words = []
        i = 0
        while i < len(words):
            add_word = True
            for j in [3,2,1]:
                if " ".join(words[i: i+j]) in filter_words:
                    i += j
                    add_word = False
                    break
            if add_word:
                filtered_words.append(words[i])
                i += 1
        if len(filtered_words):
            return text_normalizer.normalize_sentence(sentence)
        return ""

    def split_extended_contexts(self, real_sentence, intervention, context, intervention_type):
        check_values = [intervention]
        if intervention_type != "standard":
            check_values = [self.all_crops]
        right_contexts = []
        found_crops = set()
        empty_contexts = []
        for context_split in text_normalizer.split_sentence_to_parts(context, remove_and_or=True):
            nothing_found = True
            for val in check_values:
                if val in text_normalizer.normalize_sentence(context_split):
                    nothing_found = False
                    if re.search(r"\b%s"%val, text_normalizer.normalize_sentence(context_split)):
                        if intervention_type != "standard" and "product" not in val and re.search(
                            r"\b%s\b"%val, text_normalizer.normalize_sentence(context_split)):
                            found_crops.add(val)
                        right_contexts.append((val, text_normalizer.normalize_sentence(context_split).strip()))
            if nothing_found:
                empty_contexts.append(text_normalizer.normalize_sentence(context_split).strip())
            if intervention_type == "standard":
                for val in self.all_crops:
                    if val in text_normalizer.normalize_sentence(context_split) and "product" not in val:
                        if re.search(r"\b%s\b"%val, text_normalizer.normalize_sentence(context_split)):
                            found_crops.add(val)
        new_contexts = []
        for name, context in right_contexts:
            new_contexts.append(context)
            for crop in sorted(found_crops, key=lambda x: (len(x.split()), len(x)), reverse=True):
                context = re.sub(r"\b%s\b"%crop, " ## ", context)
            context = re.sub(r"(\s*##\s*)+", " ## ", context)
            for crop in found_crops:
                new_context = re.sub("##", crop, context)
                new_contexts.append(new_context)
        
        if intervention_type != "standard":
            for empty_context in empty_contexts:
                for crop in sorted(found_crops, key=lambda x: (len(x.split()), len(x)), reverse=True):
                    empty_context = re.sub(r"\b%s\b"%crop, " ## ", empty_context)
                empty_context = re.sub(r"(\s*##\s*)+", " ## ", empty_context)
                if "#" not in empty_context:
                    empty_context = "## " + empty_context
                for crop in found_crops:
                    new_context = re.sub("##", crop, empty_context)
                    new_contexts.append(new_context)
        
        full_new_contexts = []
        for context in new_contexts:
            new_context = re.sub(r"\b(nb|nb batch)$", "", context).strip()
            full_new_contexts.append(new_context)
                
        new_contexts = set(
            [self.clean_from_filter_words(context, self.stopwords_crops_normalized) for context in full_new_contexts]) - set([""])
        full_context_rows = []
        if intervention_type == "standard":
            for context in new_contexts:
                if intervention in context:
                    full_context_rows.append((intervention, context))
                else:
                    full_context_rows.append((context, context))
        else:
            for context in new_contexts:
                full_context_rows.append((context, context))
        return full_context_rows
