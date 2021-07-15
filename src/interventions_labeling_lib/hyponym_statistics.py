import math 
import pandas as pd
from utilities.excel_writer import ExcelWriter
import spacy
import re
from text_processing import search_engine_insensitive_to_spelling
from text_processing import text_normalizer
import pickle
import nltk
from time import time
import os
from sklearn.utils import shuffle

nlp = spacy.load('en_core_web_sm')

class HyponymPairInfo:
    
    def __init__(self, narrow_concept, broad_concepts):
        self.narrow_concept = narrow_concept
        self.broad_concepts = broad_concepts
        self.mentioned_together = 0
        self.narrow_concept_mentions = 0
        self.label = "No label"
        self.common_topic_frequency = 0
        self.our_data_frequency = 0
        self.words_count = 0
        self.is_abbreviation = 0
        self.abbreviations_meanings = ""
        
    def to_row(self):
        return (self.narrow_concept, ";".join(self.broad_concepts), self.narrow_concept_mentions, self.mentioned_together, \
               self.common_topic_frequency,self.our_data_frequency, self.words_count, self.is_abbreviation, self.label)

class HyponymStatistics:
    
    def __init__(self, key_word_mappings, search_engine_inverted_index, abbreviations_resolver, dict_hyponyms, hyponym_search, dict_with_synonyms = {}, filter_word_list = [], filter_hypernyms = []):
        self.total = 0
        self.hypernym_filter_words = set(key_word_mappings)
        self.hyponyms_search = hyponym_search
        self.dict_hypernym_mentions_in_inverted_index = {}
        self.dict_narrow_concepts = {}
        self.narrow_concepts_to_be_filtered = filter_hypernyms
        self.dict_hyponyms = dict_hyponyms
        self.fill_narrow_concepts()
        self.dict_with_synonyms = dict_with_synonyms
        self.fill_labels()
        self.search_engine_inverted_index = search_engine_inverted_index
        self.filter_word_list = filter_word_list
        self._abbreviations_resolver = abbreviations_resolver
    
    def fill_labels(self, filename = '../data/hyponyms_for_train.xlsx'):
        self.label_dict = {}
        if not os.path.exists(filename):
            return 
        temp_df = pd.read_excel('../data/hyponyms_for_train.xlsx').fillna("")
        for i in range(len(temp_df)):
            self.label_dict[temp_df["Narrow concept"].values[i]] = temp_df["Label"].values[i]
        
    def fill_narrow_concepts(self):
        for key in self.dict_hyponyms:
            for key_word in self.dict_hyponyms[key]:
                self.total += len(self.dict_hyponyms[key][key_word])
                if key_word not in self.dict_narrow_concepts:
                    self.dict_narrow_concepts[key_word] = {}
                if key not in self.dict_narrow_concepts[key_word]:
                    self.dict_narrow_concepts[key_word][key] = 0
                self.dict_narrow_concepts[key_word][key] += len(self.dict_hyponyms[key][key_word])
    
    def contains_filter_words(self, word):
        for filt_word in self.hypernym_filter_words:
            if filt_word in word:
                return True
        return False

    def find_mentions_in_inverted_index(self, word):
        if word in self.dict_hypernym_mentions_in_inverted_index:
            return self.dict_hypernym_mentions_in_inverted_index[word]
        self.dict_hypernym_mentions_in_inverted_index[word] = self.search_engine_inverted_index.find_articles_with_keywords([word], 1.0)
        return self.dict_hypernym_mentions_in_inverted_index[word]

    def calculate_mentions_together(self, key, key_word):
        if key not in self.dict_with_synonyms:
            return len(self.find_mentions_in_inverted_index(key).intersection(self.find_mentions_in_inverted_index(key_word)))
        total = 0
        for word in self.dict_with_synonyms[key]:
            total += len(self.find_mentions_in_inverted_index(word).intersection(self.find_mentions_in_inverted_index(key_word)))
        return total

    def find_mentions_with_synonyms(self, key):
        if key not in self.dict_with_synonyms:
            return len(self.find_mentions_in_inverted_index(key))
        total = 0
        for word in self.dict_with_synonyms[key]:
            total += len(self.find_mentions_in_inverted_index(word))
        return total
    
    def calculate_articles_with_key_word_coocurrence(self, key, key_word):
        return self.find_mentions_in_inverted_index(key).intersection(self.find_mentions_in_inverted_index(key_word))

    def calculate_frequencies(self):
        hyponym_pairs = []
        for key in self.dict_hyponyms.keys():
            if self.contains_filter_words(key):
                for key_word in self.dict_hyponyms[key]:
                    mentioned_together = self.calculate_mentions_together(key,key_word)
                    hyponym_pairs.append((key, key_word, self.find_mentions_with_synonyms(key), len(self.find_mentions_in_inverted_index(key_word)), mentioned_together, self.dict_hyponyms[key][key_word][0][0]))
        return hyponym_pairs
    
    def is_filter_word_main(self, word_expression):
        doc = nlp(word_expression)
        for chunk in doc.noun_chunks:
            if chunk.root.text in self.hypernym_filter_words:
                return True
        return False

    def separate_abbreviations_from_text(self, text):
        abbreviations = []
        word = ""
        for word_part in text.split():
            if text_normalizer.is_abbreviation(word_part):
                word = word + " " + word_part
            else:
                if word != "":
                    abbreviations.append(word.strip())
                    word = ""
        if word != "":
            abbreviations.append(word.strip())
        return abbreviations
    
    def add_resolved_abbreviations_to_text(self, text, articles_number):
        if articles_number not in self.search_engine_inverted_index.docs_with_abbreviaitons_by_id:
            return text
        abbreviations = self.separate_abbreviations_from_text(text)
        for abbreviation in abbreviations:
            if abbreviation in self.search_engine_inverted_index.docs_with_abbreviaitons_by_id[articles_number]:
                text  = text + " " + self.search_engine_inverted_index.docs_with_abbreviaitons_by_id[articles_number][abbreviation][0]
        for word_expr in self._abbreviations_resolver.sorted_words_to_abbreviations:
            if len(self._abbreviations_resolver.sorted_words_to_abbreviations[word_expr]) == 1:
                if word_expr in text and self._abbreviations_resolver.sorted_words_to_abbreviations[word_expr][0][0] not in text:
                    text  = text + " " + self._abbreviations_resolver.sorted_words_to_abbreviations[word_expr][0][0]
        return text.strip()

    def add_resolved_abbreviations_to_narrow_concept(self, text, articles_number):
        abbreviations = self.separate_abbreviations_from_text(text)
        if len(abbreviations) == 0:
            return text
        text += " # "
        abbr_meanings = []
        for abbreviation in abbreviations:
            if articles_number in self.search_engine_inverted_index.docs_with_abbreviaitons_by_id and abbreviation in self.search_engine_inverted_index.docs_with_abbreviaitons_by_id[articles_number]:
                abbr_meanings.append(self.search_engine_inverted_index.docs_with_abbreviaitons_by_id[articles_number][abbreviation][0])
            else:
                if abbreviation in self._abbreviations_resolver.sorted_resolved_abbreviations:
                    abbr_meanings.append(self._abbreviations_resolver.sorted_resolved_abbreviations[abbreviation][0][0])
        text += ";".join(abbr_meanings)
        return text
    
    def fill_narrow_dict(self):
        narrow_dict = {}
        for key in self.dict_hyponyms.keys():
            if self.contains_filter_words(key) and self.is_filter_word_main(key):
                for key_word in self.dict_hyponyms[key]:
                    articles_number = self.dict_hyponyms[key][key_word][0][1]
                    key_word_with_resolved_abbreviations = self.add_resolved_abbreviations_to_narrow_concept(key_word, articles_number)
                    if key_word_with_resolved_abbreviations not in narrow_dict:
                        narrow_dict[key_word_with_resolved_abbreviations] = []
                    narrow_dict[key_word_with_resolved_abbreviations].append(self.add_resolved_abbreviations_to_text(key, articles_number))
        return narrow_dict
    
    def get_pruned_pairs(self, narrow_dict = None):
        if narrow_dict == None:
            narrow_dict = self.fill_narrow_dict()
        hyponym_pairs = []
        for key in narrow_dict:
            info = HyponymPairInfo(key.split("#")[0], narrow_dict[key])
            info.abbreviations_meanings = key.split("#")[1] if len(key.split("#")) > 1 else ""
            mentioned_set = set()
            info.narrow_concept_mentions = len(self.find_mentions_in_inverted_index(info.narrow_concept))
            for filter_word in self.hypernym_filter_words:
                mentioned_set = mentioned_set.union(self.calculate_articles_with_key_word_coocurrence(info.narrow_concept,filter_word))
            info.mentioned_together = len(mentioned_set)
            hyponym_pairs.append(info)
        return hyponym_pairs
    
    def find_hierarchy_pairs(self):
        hyponym_pairs = []
        hierarchy = {}
        for key in self.dict_hyponyms.keys():
            if self.contains_filter_words(key):
                for key_word in self.dict_hyponyms[key]:
                    w_x_y = len(self.dict_hyponyms[key][key_word])
                    hyponym_pairs.append((key_word, key, "Directed", w_x_y))
                last_word = key.split()[-1]
                if last_word != key:
                    if last_word not in hierarchy:
                        hierarchy[last_word] = {}
                    if key not in hierarchy[last_word]:
                        hierarchy[last_word][key] = 0
                    hierarchy[last_word][key] += 1
        for key in hierarchy:
            for key_word in hierarchy[key]:
                hyponym_pairs.append((key_word, key, "Directed", hierarchy[key][key_word]))
        for key in hierarchy:
            if key != "intervention":
                hyponym_pairs.append((key, "intervention", "Directed", 1))
        return hyponym_pairs
    
    def find_inverted_hierarchy_pairs(self):
        hyponym_pairs = []
        hierarchy = {}
        for key in self.dict_hyponyms.keys():
            if self.contains_filter_words(key):
                for key_word in self.dict_hyponyms[key]:
                    w_x_y = len(self.dict_hyponyms[key][key_word])
                    hyponym_pairs.append((key, key_word, "Directed", w_x_y))
                last_word = key.split()[-1]
                if last_word != key:
                    if last_word not in hierarchy:
                        hierarchy[last_word] = {}
                    if key not in hierarchy[last_word]:
                        hierarchy[last_word][key] = 0
                    hierarchy[last_word][key] += 1
        for key in hierarchy:
            for key_word in hierarchy[key]:
                hyponym_pairs.append((key, key_word, "Directed", hierarchy[key][key_word]))
        return hyponym_pairs
    
    def save_hierarchy_pairs(self, filename):
        hyponym_pairs = self.find_hierarchy_pairs()
        ExcelWriter().save_data_in_excel(hyponym_pairs,["Source", "Target", "Type", "Weight"],filename)
    
    def save_inverted_hierarchy_pairs(self, filename):
        hyponym_pairs = self.find_inverted_hierarchy_pairs()
        ExcelWriter().save_data_in_excel(hyponym_pairs, ["Source", "Target", "Type", "Weight"],filename)
    
    def calculate_our_dataset_articles_freq(self,concept):
        score = 0
        for word in concept.split():
            articles_with_word = len(self.search_engine_inverted_index.get_articles_by_word(word))
            if articles_with_word > 0:
                score += (articles_with_word/self.search_engine_inverted_index.total_articles_number)
        return score/ len(concept.split())

    def has_noun(self, word):
        has_noun = False
        if "antibiotic" in word:
            has_noun = True
        for w in word.split():
            if "NN" in nltk.PerceptronTagger().tag([w])[0][1] or "VBG" in nltk.PerceptronTagger().tag([w])[0][1] or text_normalizer.is_abbreviation(w):
                has_noun = True
        for w in nltk.PerceptronTagger().tag(word.split()):
            if "NN" in w[1] or "VBG" in w[1] or text_normalizer.is_abbreviation(w[0]):
                has_noun = True
        return has_noun

    def has_word_with_more_than_two_letters(self, narrow_concept):
        add_to_dict = True
        is_ok = False
        for word in narrow_concept.strip().split():
            if (not text_normalizer.is_abbreviation(word) and len(word) > 2) or text_normalizer.is_abbreviation(word):
                is_ok = True
        if not is_ok:
            add_to_dict = False
        if len(narrow_concept.strip()) < 2 or (len(narrow_concept.strip()) < 3 and not text_normalizer.is_abbreviation(narrow_concept.strip())):
            add_to_dict = False
        return add_to_dict

    def has_word_with_more_than_one_nondigit_letter(self, narrow_concept):
        is_ok = False
        for word in narrow_concept.strip().split():
            if not text_normalizer.has_word_with_one_non_digit_symbol(word):
                is_ok = True
        return is_ok

    def is_word_too_common(self, narrow_concept, threshold, less_strict):
        frequent_words = 0
        add_to_dict = True
        for word in narrow_concept.split():
            if word in text_normalizer.freq_words or text_normalizer.get_rank_of_word(word) < 0.99:
                frequent_words += 1
        if len(narrow_concept.split()) == frequent_words:
            if frequent_words > 1:
                score = text_normalizer.calculate_common_topic_score(narrow_concept)
                if score <= threshold:
                    add_to_dict = False
            else:
                add_to_dict = less_strict and narrow_concept != "" and text_normalizer.calculate_common_topic_score(narrow_concept) >= threshold
        return add_to_dict

    def clean_narrow_concept(self, narrow_concept):
        new_word = ""
        for word in narrow_concept.split():
            if word not in self.filter_word_list and (text_normalizer.is_abbreviation(word) or len(word) > 1) and word not in text_normalizer.stopwords_all:
                new_word= new_word + " " + word
        narrow_concept = new_word.strip()
        if text_normalizer.is_abbreviation(narrow_concept) and narrow_concept in self._abbreviations_resolver.sorted_resolved_abbreviations:
            to_exclude = False
            for chunk in nlp(self._abbreviations_resolver.sorted_resolved_abbreviations[narrow_concept][0][0]).noun_chunks:
                if chunk.root.text in self.filter_word_list:
                    to_exclude = True
            if to_exclude:
                narrow_concept = ""
        if not self.has_noun(narrow_concept):
            narrow_concept = ""
        if narrow_concept.strip() in self.filter_word_list or narrow_concept.strip() in self.narrow_concepts_to_be_filtered:
            narrow_concept = ""
        return narrow_concept

    def clean_from_too_frequent_words(self, word_expression):
        words_to_join = []
        all_to_join = False 
        for w in word_expression.split():
            if w in text_normalizer.stopwords_all or w in ["ass", "including"]:
                continue
            if all_to_join or text_normalizer.get_rank_of_word(w) >= 0.4:
                words_to_join.append(w)
                all_to_join = True
        return " ".join(words_to_join) if len(words_to_join) > 0 else ""

    def clean_concept(self, concept, threshold = 0.4, less_strict = True):
        concept = self.clean_narrow_concept(concept)
        concept = self.clean_from_too_frequent_words(concept)
        concept = " ".join(reversed(
            self.clean_from_too_frequent_words(" ".join(reversed(concept.split()))).split()))
        if concept.strip() == "":
            return ""
        add_to_dict = self.is_word_too_common(concept, threshold, less_strict)
        if add_to_dict:
            add_to_dict = self.has_word_with_more_than_two_letters(concept)
        if add_to_dict:
            add_to_dict = self.has_word_with_more_than_one_nondigit_letter(concept)
        return concept if add_to_dict else ""
    
    def get_pruned_pairs_with_info(self, pruned_pairs, folder="../notebooks", threshold = 0.4, less_strict = False):
        hyponym_pairs = []
        cleaned_words = []
        for info in pruned_pairs:
            info.narrow_concept = self.clean_narrow_concept(info.narrow_concept)
            words_cleaned = self.clean_from_too_frequent_words(info.narrow_concept)
            if info.narrow_concept != words_cleaned:
                cleaned_words.append((info.narrow_concept, words_cleaned))
                info.narrow_concept = words_cleaned
            if info.narrow_concept.strip() == "":
                continue
            if not self.has_noun(info.narrow_concept):
                cleaned_words.append((info.narrow_concept, info.narrow_concept))
                info.narrow_concept = ""
            if len(info.narrow_concept.split()) == 2 and info.narrow_concept.split()[-1].strip() == "farmer":
                info.narrow_concept = ""
            if info.narrow_concept == "":
                continue
            add_to_dict = self.is_word_too_common(info.narrow_concept, threshold, less_strict)
            if add_to_dict:
                add_to_dict = self.has_word_with_more_than_two_letters(info.narrow_concept)
            if add_to_dict:
                add_to_dict = self.has_word_with_more_than_one_nondigit_letter(info.narrow_concept)
            if info.narrow_concept_mentions == 0:
                add_to_dict = False
            if add_to_dict:
                    if info.narrow_concept in self.label_dict:
                        info.label = self.label_dict[info.narrow_concept]
                    if info.abbreviations_meanings.strip() != "":
                        info.narrow_concept = info.narrow_concept + " # " + info.abbreviations_meanings.strip()
                    info.common_topic_frequency = round(text_normalizer.calculate_common_topic_score(info.narrow_concept),4)
                    info.our_data_frequency = round(self.calculate_our_dataset_articles_freq(info.narrow_concept),4)
                    info.words_count = len(info.narrow_concept.split())
                    info.is_abbreviation = int(text_normalizer.is_abbreviation(info.narrow_concept))
                    hyponym_pairs.append(info.to_row())
        ExcelWriter().save_data_in_excel(cleaned_words, ["old word","new word"], os.path.join(folder, "cleaned_words %d.xlsx"%time()))
        return hyponym_pairs
        
    def save_pruned_hyponym_pairs_to_file(self, filename, folder="../notebooks",threshold = 0.4):
        hyponym_pairs = self.get_pruned_pairs_with_info(self.get_pruned_pairs(), folder, threshold = threshold)
        ExcelWriter().save_data_in_excel(hyponym_pairs, ["Narrow concept", "Broad concepts", "narrow concept mentioned in articles", "broad concept and narrow concept occured together in articles",\
                                                  "common topic frequency","our dataset word weight","Words count","Is abbreviation", "Label"],filename, column_probabilities=['narrow concept mentioned in articles',
       'broad concept and narrow concept occured together in articles',
       'common topic frequency', 'our dataset word weight'], column_interventions=['Label'], column_outliers=[("Words count", 1),("Is abbreviation", 1)])

    def save_pruned_hyponym_pairs_to_file_for_test(self, filename, size=300):
        hyponym_pairs = shuffle([row for row in self.get_pruned_pairs_with_info(self.get_pruned_pairs()) if row[-1] =="No label"])[:size]
        ExcelWriter().save_data_in_excel(hyponym_pairs, ["Narrow concept", "Broad concepts", "narrow concept mentioned in articles", "broad concept and narrow concept occured together in articles",\
                                                  "common topic frequency","our dataset word weight","Words count","Is abbreviation","Label"],filename, column_probabilities=['narrow concept mentioned in articles',
       'broad concept and narrow concept occured together in articles',
       'common topic frequency', 'our dataset word weight'], column_interventions=['Label'],column_outliers=[("Words count", 1),("Is abbreviation", 1)])

    
    def save_hyponym_pairs_to_file(self, filename):
        hyponym_pairs = self.calculate_frequencies()
        ExcelWriter().save_data_in_excel(hyponym_pairs, ["Broad concept", "Narrow concept", "broad concept mentioned in articles", "narrow concept mentioned in articles", "broad concept and narrow concept occured together in articles",\
                                                      "Example of sentence"],filename,column_probabilities=["broad concept mentioned in articles", "narrow concept mentioned in articles", "broad concept and narrow concept occured together in articles"])