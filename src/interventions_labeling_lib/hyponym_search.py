from interventions_labeling_lib.hearst_pattern_finder import HearstPatterns
from text_processing import text_normalizer
import pickle
from time import time
from text_processing import concepts_merger
import os

class HyponymsSearch:
    
    def __init__(self):
        self.symbols_count = 5
        self.dict_hyponyms = {}
        self.global_hyponyms = {}
        self.concepts_merger = concepts_merger.ConceptsMerger(self.symbols_count)
    
    def add_hyponyms(self, hyponyms, article_number):
        if article_number not in self.global_hyponyms:
            self.global_hyponyms[article_number] = []
        self.global_hyponyms[article_number].extend(hyponyms)
        for detected_hyponym in hyponyms:
            for word in detected_hyponym[:2]:
                if word.strip() != "":
                    self.concepts_merger.add_item_to_dict(word,article_number)
    
    def add_hyponyms_to_dict(self):
        self.dict_hyponyms = {}
        for article_number in self.global_hyponyms:
            for detected_hyponym in self.global_hyponyms[article_number]:
                if detected_hyponym[1] in self.concepts_merger.new_mapping and detected_hyponym[0] in self.concepts_merger.new_mapping:
                    first_word = self.concepts_merger.new_mapping[detected_hyponym[1]]
                    second_word = self.concepts_merger.new_mapping[detected_hyponym[0]]
                    if first_word not in self.dict_hyponyms:
                        self.dict_hyponyms[first_word] = {}
                    if second_word not in self.dict_hyponyms[first_word]:
                        self.dict_hyponyms[first_word][second_word] = []
                    self.dict_hyponyms[first_word][second_word].append((detected_hyponym[2].replace("NP_","").replace("_"," "), article_number, detected_hyponym[3]))
        self.concepts_merger = concepts_merger.ConceptsMerger(self.symbols_count)
    
    def create_hyponym_dict(self, search_engine_inverted_index, threshold = 0.92):
        self.concepts_merger.merge_concepts(search_engine_inverted_index, threshold)
        self.add_hyponyms_to_dict()
    
    def find_hyponyms_and_hypernyms(self, articles_df, search_engine_inverted_index, filename_with_data="", columns_to_use=["title", "abstract"]):
        if filename_with_data != "" and os.path.exists(filename_with_data):
            hyp_s = pickle.load(open(filename_with_data,"rb"))
            for article in hyp_s.global_hyponyms:
                self.add_hyponyms(hyp_s.global_hyponyms[article], article)
        h = HearstPatterns(True)
        for i in range(len(articles_df)):
            art_id = articles_df["id"].values[i] if "id" in articles_df.columns else i
            if art_id in self.global_hyponyms:
                continue
            if i % 5000 == 0 or i == len(articles_df) - 1:
                print("Processed %d articles" % i)
            text = ""
            for column in columns_to_use:
                text = text + " . " + articles_df[column].values[i]
            self.add_hyponyms(h.find_hyponyms(text), art_id)
        self.create_hyponym_dict(search_engine_inverted_index)