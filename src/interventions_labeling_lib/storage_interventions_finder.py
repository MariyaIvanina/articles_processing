from allennlp.predictors.predictor import Predictor
from text_processing import text_normalizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from text_processing import geo_names_finder
from geotext import GeoText
import pandas as pd
import re
import spacy
from utilities import excel_writer
from interventions_labeling_lib import hyponym_search
from interventions_labeling_lib import hearst_pattern_finder
from interventions_labeling_lib import hyponym_statistics
import os
from text_processing import abbreviations_resolver
from time import time

lmtzr = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

class StorageInterventionsFinder:

    def __init__(self, search_engine_inverted_index, _abbreviations_resolver,  folder, words_preventions=[],filter_word_list = [], filter_hypernyms = []):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.all_crops = text_normalizer.build_filter_dictionary([ "../data/map_plant_products.xlsx", "../data/map_animal_products.xlsx"])
        self.words_preventions = words_preventions
        self.words_storage_indicators = ["store", "seal","reuse","immerse"]
        self.countries_finder = geo_names_finder.GeoNameFinder()
        self.container_words = ["bag","container","tray","sack","pod","bin","storage","postharvest","post harvest"]
        self.frequent_words = self.get_frequent_words()
        self.filter_word_list = filter_word_list
        self.filter_hypernyms = filter_hypernyms
        self.folder = folder
        self.hearst_pattern_finder = hearst_pattern_finder.HearstPatterns(True)
        self._abbreviations_resolver = _abbreviations_resolver
        self.parsed_sentences = {}
        self.search_engine_inverted_index = search_engine_inverted_index

    def get_frequent_words(self):
        freq_words = set()
        temp_df = pd.read_excel('../data/MostFrequentWords.xlsx').fillna("")
        for i in range(len(temp_df)):
            word = temp_df["Word"].values[i].lower().strip() 
            freq_words.add(word)
        return freq_words

    def parse_sentences(self, articles_df,start_index = 0, finish_index = None):
        start = start_index
        start_id_assigned = False
        all_keywords = set(self.words_storage_indicators).union(self.container_words)
        cnt_processed = 0
        parsed_sent_temp = {}
        for i in sorted(list(self.search_engine_inverted_index.find_articles_with_keywords(list(all_keywords)))):
            art_id = articles_df["id"].values[i] if "id" in articles_df.columns else i
            if art_id < start_index:
                continue
            if art_id in self.parsed_sentences:
                continue
            if not start_id_assigned:
                start_id_assigned = True
                start = i
            cnt_processed += 1
            if cnt_processed % 500 == 0:
                print("Processed %d articles"% cnt_processed)
            self.parsed_sentences[art_id] = []
            title = text_normalizer.remove_accented_chars(articles_df["title"].values[i].lower()\
                if articles_df["title"].values[i].isupper() else articles_df["title"].values[i])
            sentences = [title]
            sentences.extend(nltk.sent_tokenize(text_normalizer.remove_accented_chars(articles_df["abstract"].values[i])))
            for sent in sentences:
                res = self.predictor.predict(sentence=sent)
                if "verbs" in res:
                    for verb_info in res["verbs"]:
                        if verb_info["verb"] not in text_normalizer.stopwords_all and lmtzr.lemmatize(verb_info["verb"]) not in text_normalizer.stopwords_all:
                            self.parsed_sentences[art_id].append((verb_info["verb"], verb_info["description"]))
            parsed_sent_temp[art_id] = self.parsed_sentences[art_id]
            if finish_index != None and art_id > finish_index:
                self.save_parsed_data(parsed_sent_temp, articles_df, start, finish_index)
                break
            if cnt_processed % 3000 == 0:
                self.save_parsed_data(parsed_sent_temp, articles_df, start, i)
                parsed_sent_temp = {}
                start = i
        if len(parsed_sent_temp) > 0:
            self.save_parsed_data(parsed_sent_temp, articles_df, start, i)
        print("Finished parsing sentences")

    def save_parsed_data(self, parsed_sentences, articles_df, start_index, finish_index):
        data_to_save = []
        for i in range(start_index, finish_index+1):
            art_id = articles_df["id"].values[i] if "id" in articles_df.columns else i
            if not art_id in parsed_sentences:
                continue
            for d in parsed_sentences[art_id]:
                data_to_save.append((art_id, articles_df["title"].values[i], d[0], d[1]))

        excel_writer.ExcelWriter().save_data_in_excel(data_to_save, ["DocId","Title", "Verb", "Tagged sentences"], \
            os.path.join(self.folder,"%d-%d_parsed_%d.xlsx"%(start_index, finish_index, time())))

    def load_parsed_sentences(self):
        self.parsed_sentences = {}
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        for filename in os.listdir(self.folder):
            temp_df = pd.read_excel(os.path.join(self.folder, filename))
            for i in range(len(temp_df)):
                if temp_df["DocId"].values[i] not in self.parsed_sentences:
                    self.parsed_sentences[temp_df["DocId"].values[i]] = []
                self.parsed_sentences[temp_df["DocId"].values[i]].append((temp_df["Verb"].values[i], temp_df["Tagged sentences"].values[i]))
        print("Loaded parsed sentences")

    def split_sentence(self, sentence):
        sent_parts = []
        for m in re.finditer("\[.*?\]", sentence):
            if "ARGM-LOC:" in m.group(0):
                word = m.group(0).split(":")[1].strip()[:-1]
                if self.is_proper_intervention(word):
                    sent_parts.append(m.group(0))
            elif re.search("ARG\d+:", m.group(0)) != None:
                word = m.group(0).split(":")[1].strip()[:-1]
                if text_normalizer.contain_full_name(word, self.container_words):
                    sent_parts.append(m.group(0))
        return sent_parts

    def is_proper_intervention(self, word):
        return word not in text_normalizer.stopwords_all and lmtzr.lemmatize(word) not in text_normalizer.stopwords_all and len(self.countries_finder.get_all_countries_from_text(word)[0]) == 0\
                and len(GeoText(text_normalizer.normalize_without_lowering(word)).country_mentions) == 0 and word not in self.all_crops and lmtzr.lemmatize(word) not in self.all_crops

    def split_sentence_for_preventions(self, sentence):
        sent_parts = []
        for m in re.finditer("\[.*?\]", sentence):
            if "ARG0:" in m.group(0):
                word = m.group(0).split(":")[1].strip()[:-1]
                if self.is_proper_intervention(word):
                    if text_normalizer.contain_full_name(word, self.container_words):
                        sent_parts.append(m.group(0))
        return sent_parts

    def normalize_interventions(self, text):
        words = []
        for w in text.split():
            if re.match(".*\d+.*", w) == None and re.match("\W+",w) == None and w not in text_normalizer.stopwords_all and \
            lmtzr.lemmatize(w) not in text_normalizer.stopwords_all and w.lower() not in text_normalizer.stopwords_all:
                for w_tag in nlp(w):
                    if w_tag.tag_ not in ["CD"]:
                        words.append(lmtzr.lemmatize(w))
        return " ".join(words)

    def lower_sentence(self, text):
        words = []
        for word in text.split():
            if text_normalizer.is_abbreviation(word):
                words.append(text_normalizer.normalize_abbreviation(word))
            else:
                words.append(word.lower())
        return " ".join(words).replace("-","").replace("+","")

    def process_extracted_info(self, extracted_info):
        processed_info = []
        for word_expr in extracted_info:
            word = re.sub("\s+", " ", re.sub("\(.*?\)"," ",word_expr.split(":")[1].strip()[:-1]))
            word_chunks = [self.hearst_pattern_finder.clean_hyponym_term(w) for w in self.hearst_pattern_finder.replace_np_sequences(self.lower_sentence(word)).split() if w.startswith("NP_")]
            for w in word_chunks:
                normalized_text = self.normalize_interventions(w)
                if len(normalized_text.split()) > 0 and (normalized_text.split()[-1] in ["layer", "air","condition","area","temperature","content","humidity","moisture","temp",\
                                         "compartment","region","cultivar","leaf","heap","form","level","design","problem", "farm","grower","quantity","tropic","use"] \
                                         or normalized_text.split()[-1]  in self.filter_word_list):
                        normalized_text = ""
                if normalized_text in self.frequent_words or len(normalized_text) < 2 or not self.is_proper_intervention(normalized_text):
                    normalized_text = ""
                if normalized_text != "":
                    processed_info.append(normalized_text)
                if text_normalizer.contain_full_name(normalized_text, self.container_words):
                    processed_info.append(normalized_text)
        return list(set(processed_info))

    def find_interventions(self):
        print("Started finding interventions")
        new_data = []
        container_set = {}
        for doc_id in self.parsed_sentences:
            for pair in self.parsed_sentences[doc_id]:
                if type(pair[0]) != str:
                    continue
                extracted_info = []
                if text_normalizer.contain_verb_form(pair[0],self.words_preventions):
                    extracted_info = self.split_sentence_for_preventions(pair[1])
                elif text_normalizer.contain_verb_form(pair[0],self.words_storage_indicators):
                    extracted_info = self.split_sentence(pair[1])
                processed_info = ";".join(self.process_extracted_info(extracted_info))
                if processed_info != "":
                    new_data.append((pair[0], processed_info, ";".join(extracted_info), pair[1]))
                    for w in processed_info.split(";"):
                        if w not in container_set:
                            container_set[w] = []
                        container_set[w].append(doc_id)
        print("Finished finding interventions")
        return new_data, container_set

    def save_found_interventions(self, filename, search_engine_inverted_index, key_word_mappings, folder="../notebooks", save_debug_info=False):
        new_data, container_set = self.find_interventions()
        
        if save_debug_info:
            excel_writer.ExcelWriter().save_data_in_excel(new_data, ["Verb","Found interventions", "Extracted tags","Full sentence"], os.path.join(folder,"results_temp_storage.xlsx"))

        hyp_s = hyponym_search.HyponymsSearch()
        h_s = hearst_pattern_finder.HearstPatterns(True)
        for container_hyp in container_set:
            hyp_s.add_hyponyms([(h_s.clean_hyponym_term(container_hyp),h_s.clean_hyponym_term(container_hyp)+" ; storage intervention", "",0)], container_set[container_hyp][0])
        print("Added hyponyms to dict")
        hyp_s.create_hyponym_dict(search_engine_inverted_index)
        print("Started hyponyms statistics calculation")
        hyp_s_pruned = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index, self._abbreviations_resolver, hyp_s.dict_hyponyms,hyp_s, filter_word_list=self.filter_word_list, filter_hypernyms = self.filter_hypernyms)
        hyp_s_pruned.save_pruned_hyponym_pairs_to_file(filename, folder, threshold = 0.2)
        print("Finished hyponyms statistics calculation")