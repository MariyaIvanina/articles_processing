import os
import pandas as pd
from text_processing import text_normalizer
import pandas as pd
import nltk
import re
from text_processing import concepts_merger
import gensim
from utilities import excel_writer
from utilities import excel_reader
from gensim.models.phrases import Phrases, Phraser

class InterventionsDictionaryDeriver:

    def __init__(self, middle_layer_keywords = None, google_model_dir = "", phraser_file = "../model/phrases_bigram.model", intervention_words=None):
        if intervention_words is None:
            self.intervention_words = [
                "practice", "approach", "intervention", "input","strategy", "policy", "program", "programme", "initiative", "technology",
                "science", "technique", "innovation", "biotechnology","machine", "mechanism", "equipment","device","project"
            ]
        if middle_layer_keywords is None:
            self.middle_layer_keywords = {
            "Technology intervention": ["soil", "fertilizer", "water", "irrigation", "tillage", "genetic", "seed", "crop", "breeding", "milking", "energy", "health","nutrition","weather", "mobile"],
            "Ecosystem intervention": ["water", "conservation", "forest", "environment", "land", "climate", "agroforestry", "energy", "emission", "recycling", "ecosystem"],
            "Socioeconomic intervention": ["finance", "market", "price", "social", "economic", "health", "livelihood", "rural", "community", "education", "subsidy", "food", "incentive", "government"],
            "Storage intervention": ["storage", "technology", "packaging"],
            "Mechanisation intervention": ["machine", "device", "tractor", 'equipment', "power"]
            }
        else:
            self.middle_layer_keywords = middle_layer_keywords
        if google_model_dir != "":
            self.google_model = gensim.models.Word2Vec.load(os.path.join(google_model_dir, os.path.basename(google_model_dir) + ".model"))
        self.phrases = Phraser.load(phraser_file)

    def process_interventions_df(self, interventions_df):
        concepts_to_delete = set()
        for i in range(len(interventions_df)):
            narrow_concept = re.sub(r"\bprogramme\b", "program", interventions_df["Narrow concept"].values[i])
            narrow_concept = narrow_concept.split("#")[0].strip()
            narrow_concept = " ".join(text_normalizer.get_stemmed_words_inverted_index(narrow_concept))
            abbreviation_meanings = interventions_df["Narrow concept"].values[i].split("#")[1].strip()\
                if len(interventions_df["Narrow concept"].values[i].split("#")) > 1 else ""
            if interventions_df["Predicted Label"].values[i] != "Non-intervention" and interventions_df["Label"].values[i] != "Non-intervention":
                if narrow_concept not in self.interventions_set:
                    self.interventions_set[narrow_concept] = {}
                self.interventions_set[narrow_concept][abbreviation_meanings] = interventions_df["Predicted Label"].values[i]
            else:
                concepts_to_delete.add((narrow_concept, abbreviation_meanings))
        return concepts_to_delete

    def create_dictionary(self, search_engine_inverted_index, _abbreviations_resolver,
            folder = None, interventions_df=None, threshold = 0.91):
        self.interventions_set = {}
        concepts_to_delete = set()
        _excel_reader = excel_reader.ExcelReader()
        if folder is not None:
            for doc_name in os.listdir(folder):
                interventions_df = _excel_reader.read_df_from_excel(os.path.join(folder, doc_name))
                concepts_to_delete = concepts_to_delete.union(self.process_interventions_df(interventions_df))
        elif interventions_df is not None:
            concepts_to_delete = self.process_interventions_df(interventions_df)
            
        self.delete_non_interventions(concepts_to_delete)
        self.merge_interventions(search_engine_inverted_index, threshold)
        self.remove_intervention_synonyms_from_concepts(_abbreviations_resolver)
        self.remove_too_abstract_words(_abbreviations_resolver)
        self.filter_intervention_words()
        self.delete_non_interventions(concepts_to_delete)
        self.find_middle_layer_names(_abbreviations_resolver)

    def delete_non_interventions(self, concepts_to_delete):
        for narrow_concept, abbreviation_meanings in concepts_to_delete:
            if narrow_concept in self.interventions_set and abbreviation_meanings in self.interventions_set[narrow_concept]:
                del self.interventions_set[narrow_concept][abbreviation_meanings]
        for key in set(self.interventions_set):
            if len(self.interventions_set[key]) == 0:
                del self.interventions_set[key]

    def save_dictionary(self, filename):
        pairs_to_save = []
        for narrow_concept in self.new_interventions_set:
            for abbr_meaning in self.new_interventions_set[narrow_concept]:
                pairs_to_save.append((narrow_concept, abbr_meaning,\
                 self.new_interventions_set[narrow_concept][abbr_meaning][0], ";".join(list(self.new_interventions_set[narrow_concept][abbr_meaning][1]))))
        excel_writer.ExcelWriter().save_data_in_excel(pairs_to_save, ["Narrow concept", "Abbreviation meaning", "Intervention class", "Middle Layer names"], filename)

    def prepare_word_dictionary(self, _abbreviations_resolver):
        new_dict = {}
        for nar_conc in self.interventions_set:
            for abbr_meaning in self.interventions_set[nar_conc]:
                full_name = _abbreviations_resolver.replace_abbreviations(nar_conc, abbr_meaning)
                if full_name not in new_dict:
                    new_dict[full_name] = []
                new_dict[full_name].append((nar_conc, abbr_meaning))
        return new_dict

    def filter_intervention_words(self):
        new_dict = set()
        for nar_conc in self.interventions_set:
            new_dict.add(nar_conc)
        for w in new_dict:
            if w in self.intervention_words:
                del self.interventions_set[w]

    def remove_too_abstract_words(self, _abbreviations_resolver):
        new_dict = set()
        for nar_conc in self.interventions_set:
            new_dict.add(nar_conc)
        for w in new_dict:
            old_w = w
            while w.strip() != "" and w.split()[0] in ['adoption', "level", "hectare", "exterior"]:
                w = " ".join(w.split()[1:])
            if old_w != w:
                if w not in self.interventions_set and w != "":
                    self.interventions_set[w] = self.interventions_set[old_w]
                del self.interventions_set[old_w]

    def merge_interventions(self, search_engine_inverted_index, threshold = 0.92):
        _concepts_merger = concepts_merger.ConceptsMerger(5)
        for word in self.interventions_set:
            _concepts_merger.add_item_to_dict(word,0)
        _concepts_merger.merge_concepts(search_engine_inverted_index, threshold)
        new_interventions_dict = {}
        for word in _concepts_merger.inverted_dictionary:
            word_to_use = _concepts_merger.get_frequent_word(word, search_engine_inverted_index)
            new_interventions_dict[word_to_use] = self.interventions_set[word]
        self.interventions_set = new_interventions_dict

    def remove_intervention_synonyms_from_concepts(self, _abbreviations_resolver):
        new_dict = self.prepare_word_dictionary(_abbreviations_resolver)
        for w in new_dict:
            firstpart , last_part = " ".join(w.split()[:-1]), w.split()[-1]
            if firstpart in new_dict and last_part in self.google_model and (self.google_model.similarity(last_part, "approach") > 0.3 or self.google_model.similarity(last_part, "intervention") > 0.3):
                for info in new_dict[w]:
                    del self.interventions_set[info[0]][info[1]]
                    if len(self.interventions_set[info[0]]) == 0:
                        del self.interventions_set[info[0]]

    def split_words_for_dict(self, interv_class, _abbreviations_resolver):
        technology_dict_nar = {}
        for narrow_concept in self.interventions_set:
            for abbr_meaning in self.interventions_set[narrow_concept]:
                if self.interventions_set[narrow_concept][abbr_meaning] == interv_class:
                    narrow_concept_to_check = _abbreviations_resolver.replace_abbreviations(narrow_concept, abbr_meaning).replace("("," ").replace(")"," ")
                    for broad_con in text_normalizer.get_bigrams(narrow_concept_to_check, self.phrases):
                        if broad_con.strip() not in technology_dict_nar:
                            technology_dict_nar[broad_con.strip()] = set()
                        technology_dict_nar[broad_con.strip()].add((narrow_concept, abbr_meaning, narrow_concept_to_check))
        dict_for_interv_keywords = {}
        for word in self.middle_layer_keywords[interv_class]:
            dict_for_interv_keywords[word] = []
        dict_for_interv_keywords["other"] = []
        return technology_dict_nar, dict_for_interv_keywords

    def assign_middle_layer_words(self, dict_with_concepts):
        dict_w = {}
        for concept in dict_with_concepts:
            for w in dict_with_concepts[concept]:
                if w not in dict_w:
                    dict_w[w] = set()
                dict_w[w].add(concept)

        for w in dict_w:
            if len(dict_w[w]) > 1 and "other" in dict_w[w]:
                dict_w[w].remove("other")

        cnt = 0
        for w in dict_w:
            if len(dict_w[w]) > 1:
                full_res = []
                for middle_w in dict_w[w]:
                    words_in_concept = [wf for wf in text_normalizer.get_bigrams(w[2], self.phrases) if wf in self.google_model]
                    if len(words_in_concept) > 0:
                        res = self.google_model.wv.most_similar_to_given(middle_w, words_in_concept)
                        full_res.append((middle_w,  self.google_model.similarity(middle_w, res)))
                    else:
                        full_res.append((middle_w,  1.0))
                full_res = sorted(full_res, key = lambda x:x[1], reverse = True)
                if full_res[0][1] >= 0.99:
                    dict_w[w] = set([wq[0] for wq in full_res if wq[1] >= 0.99])
                else:
                    dict_w[w] = set([full_res[0][0]])
                cnt += 1
        print("Number of intervention with several classes before resolving: ", cnt)
        cnt = 0
        for w in dict_w:
            if len(dict_w[w]) > 1:
                cnt += 1
        print("Number of intervention with several classes after resolving: ",cnt)
        return dict_w

    def find_middle_layer_names(self, _abbreviations_resolver):
        self.new_interventions_set = {}
        for interv_class in self.middle_layer_keywords:
            print(interv_class)
            technology_dict_nar, dict_for_interv_keywords = self.split_words_for_dict(interv_class, _abbreviations_resolver)
            for w in technology_dict_nar:
                if w in self.google_model:
                    chosen_word = self.google_model.wv.most_similar_to_given(w, self.middle_layer_keywords[interv_class])
                    if self.google_model.similarity(w, chosen_word) < 0.3:
                        dict_for_interv_keywords["other"].append(w)
                    else:
                        dict_for_interv_keywords[chosen_word].append(w)
            dict_with_concepts = {}
            for word in dict_for_interv_keywords:
                dict_with_concepts[word] = set()
                for w in dict_for_interv_keywords[word]:
                    dict_with_concepts[word] = dict_with_concepts[word].union(technology_dict_nar[w])
                print(word, len(dict_with_concepts[word]))

            dict_w = self.assign_middle_layer_words(dict_with_concepts)
            
            for narrow_concept, abbr_meaning, narrow_concept_to_check in dict_w:
                if narrow_concept not in self.new_interventions_set:
                    self.new_interventions_set[narrow_concept] = {}
                self.new_interventions_set[narrow_concept][abbr_meaning] = (self.interventions_set[narrow_concept][abbr_meaning], dict_w[(narrow_concept, abbr_meaning, narrow_concept_to_check)])