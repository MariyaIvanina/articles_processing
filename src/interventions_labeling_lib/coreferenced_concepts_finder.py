from allennlp.predictors.predictor import Predictor
from interventions_labeling_lib.hearst_pattern_finder import HearstPatterns
from interventions_labeling_lib.hyponym_search import HyponymsSearch
from text_processing import text_normalizer
import spacy
import os
import pickle
from time import time
import langdetect

nlp = spacy.load('en_core_web_sm')

class InterventionsPairsSearch:
    
    def __init__(self, key_word_mappings):
        self.key_word_mappings = key_word_mappings
    
    def create_pairs(self, sentences):
        res = []
        for expressions,sent in sentences:
            interventions = []
            narrow_concepts = []
            for word in expressions:
                is_intervention = False
                for key_word in self.key_word_mappings:
                    if key_word in word:
                        is_intervention = True
                if is_intervention:
                    interventions.append(word)
                narrow_concepts.append(word)
            if len(interventions) > 0:
                res.append([(narrow_concept, ";".join(interventions),sent,0) for narrow_concept in narrow_concepts])
        return res

class CoreferencedConceptsFinder:

    def __init__(self, key_word_mappings):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
        self.interventions_pairs_search = InterventionsPairsSearch(key_word_mappings)
        self.hyponyms_search_part = HyponymsSearch()
        self.hearst_pattern_finder_part = HearstPatterns(True)

    def is_proper_noun(self, text):
        for word in text.split():
            if not text_normalizer.is_abbreviation(word) and word[0].isupper():
                return False
        return True

    def normalize_coreferenced_words(self, text):
        final_res =" ".join([w for w in text.split() if w not in text_normalizer.stopwords_all and w.lower() not in text_normalizer.stopwords_all])
        if not self.is_proper_noun(final_res):
            return ""
        final_res = self.hearst_pattern_finder_part.clean_hyponym_term(text)
        return final_res
        allowed_tags =  ["JJ", "JJR", "JJS", "NNS", "NN", "NNP", "NNPS", "PDT", "PRP", "PRP$", "VBG","DT"]
        has_allowed_tags = False
        for word in nlp(final_res):
            if word.tag_ in allowed_tags:
                has_allowed_tags = True
                break
        return final_res if has_allowed_tags else ""

    def parse_coreferences(self, articles_df, start_index = 0, continue_extract = False, folder = "", column_to_use="abstract"):
        if not continue_extract or folder == "":
            self.cluster_dict = {}
        else:
            self.load_parsed_coreferences(folder)
        cluster_temp = {}
        for i in range(start_index, len(articles_df)):
            art_id = articles_df["id"].values[i] if "id" in articles_df.columns else i
            abstract = text_normalizer.remove_accented_chars(articles_df[column_to_use].values[i])

            if art_id in self.cluster_dict or abstract == "":
                continue
            try:
                lang_text = langdetect.detect(articles_df[column_to_use].values[i])
                if lang_text != 'en':
                    cluster_temp[art_id] = []
                    continue
            except:
                pass
            
            if i % 3000 == 0 or i == len(articles_df) -1:
                print("%d articles are processed"%i)

            self.cluster_dict[art_id] = []
            try:
                res = self.predictor.predict(
                  document = abstract
                )
            except:
                print(i)
                cluster_temp[art_id] = self.cluster_dict[art_id]
                continue
            
            for cluster in res["clusters"]:
                cluster_processed = []
                for word_range in cluster:
                    word_expr = " ".join(res["document"][word_range[0]:word_range[1]+1])
                    cluster_processed.append(word_expr)
                self.cluster_dict[art_id].append(cluster_processed)
            cluster_temp[art_id] = self.cluster_dict[art_id]
        if len(cluster_temp) > 0:
            if not os.path.exists(folder):
                os.makedirs(folder)
            pickle.dump(cluster_temp, open(os.path.join(folder, "%d_%d.pickle"%(min(cluster_temp.keys()), time())),"wb"))

    def load_parsed_coreferences(self, folder):
        self.cluster_dict = {}
        for filename in os.listdir(folder):
            temp_dict = pickle.load(open(os.path.join(folder, filename),"rb"))
            for doc_id in temp_dict:
                self.cluster_dict[doc_id] = temp_dict[doc_id]

    def find_coreferenced_pairs(self, articles_df, search_engine_inverted_index, continue_extract = False, folder = "",
            file_with_gl_hyp = "../model/coref_hyponyms_gl.pickle", column_to_use="abstract"):
        self.parse_coreferences(articles_df, continue_extract = continue_extract, folder = folder, column_to_use=column_to_use)
        print("Parsed documents")
        global_hyponyms = pickle.load(open(file_with_gl_hyp,"rb")) if os.path.exists(file_with_gl_hyp) else {}
        for article in global_hyponyms:
            self.hyponyms_search_part.add_hyponyms(global_hyponyms[article],article)
        for i in sorted(self.cluster_dict):
            if i in self.hyponyms_search_part.global_hyponyms:
                continue
            if i % 3000 == 0 or i == max(self.cluster_dict):
                print("%d articles are processed"%i)
            set_pairs = set()
            cluster_res = []
            for cluster in self.cluster_dict[i]:
                cluster_processed = []
                for word_expr in cluster:
                    doc = nlp(word_expr)
                    final_res = self.normalize_coreferenced_words(word_expr)
                    root = [token for token in doc if token.head == token and "VB" not in token.tag_]
                    root = root[0].text if len(root) > 0 else ""
                    for word in doc.noun_chunks:
                        if root == "" or root in word.text:
                            final_res = self.normalize_coreferenced_words(word.text)
                            break
                    if final_res != "":
                        cluster_processed.append(final_res)
                if len(cluster_processed) > 0:
                    cluster_res.append(list(set(cluster_processed)))
            res_pairs = []
            for cluster in cluster_res:
                pairs = self.interventions_pairs_search.create_pairs([(cluster,"")])
                if len(pairs) > 0:
                    pairs = pairs[0]
                for pair in pairs:
                    if pair[0] not in text_normalizer.stopwords_all:
                        res_pairs.append(pair)
            self.hyponyms_search_part.add_hyponyms(res_pairs,i)
        self.hyponyms_search_part.create_hyponym_dict(search_engine_inverted_index)
        pickle.dump(self.hyponyms_search_part.global_hyponyms, open(file_with_gl_hyp,"wb"))