from itertools import combinations
import numpy as np
from scipy import spatial
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import matplotlib.pyplot as plt
from datetime import datetime
import gensim
from gensim.models.phrases import Phrases, Phraser
from text_processing import text_normalizer
import os
import re
from utilities import utils

class TopicEvaluator:

    def __init__(self, topic_modeler, folder = "../model/synonyms_retrained"):
        self.topic_modeler = topic_modeler
        self.phrases_3gram = Phraser.load(os.path.join(folder, "phrases_bigram.model"))
        self.google_2_and_3_bigrams_model = gensim.models.Word2Vec.load(os.path.join(folder, "google_plus_our_dataset/", "google_plus_our_dataset.model"))

    def get_mean_vector(self, key_word):
        key_word_embedding = np.zeros(300)
        all_words_cnt = 0
        for phr in self.phrases_3gram[text_normalizer.get_stemmed_words_inverted_index(key_word)]:
            if phr.replace("_"," ") in self.google_2_and_3_bigrams_model.wv:
                key_word_embedding += self.google_2_and_3_bigrams_model.wv[phr.replace("_"," ")]
                all_words_cnt += 1
        if all_words_cnt > 0:
            key_word_embedding /= all_words_cnt
        return key_word_embedding

    def calculate_coherence(self, term_rankings):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            pair_scores = []
            for pair in combinations( term_rankings[topic_index], 2 ):
                res = (1 - spatial.distance.cosine(self.get_mean_vector(pair[0]), self.get_mean_vector(pair[1]))) 
                if res == res:
                    pair_scores.append(res)
            topic_score = sum(pair_scores) / len(pair_scores) if len(pair_scores) > 0 else 0
            overall_coherence += topic_score
        return overall_coherence / len(term_rankings)

    def get_top_words(self, nmf_topics, n_components, n_top_words):
        tf_feature_names_topics = self.topic_modeler.tf_vectorizer_for_topics.get_feature_names()
        all_topics = []
        for i in range(n_components):
            all_topics.append([tf_feature_names_topics[w] for w  in nmf_topics.components_[i].argsort()[::-1]][:n_top_words])
        return all_topics

    def calculate_scores_for_nmf(self, n_comp_values= [15, 25, 50,75,100,125,150]):
        results = []
        for n_comp in n_comp_values:
            print(n_comp)
            nmf_topics = NMF(n_components=n_comp,
                                            random_state=0)
            t0 = time()
            p_topic = nmf_topics.fit_transform(self.topic_modeler.tf_topics)
            topic_words = self.get_top_words(nmf_topics, n_comp, 10)
            results.append((n_comp, topic_words))
            print(time() - t0)
        k_values = []
        coherences = []
        for n_comp, topic_words in results:
            k_values.append( n_comp )
            coherences.append( self.calculate_coherence( topic_words ) )
            print("K=%02d: Coherence=%.4f" % ( n_comp, coherences[-1] ) )
        return results, k_values, coherences

    def show_coherence_plot(self, k_values, coherences):
        fig, ax= plt.subplots(figsize = (5,5))
        ax.set_xlabel("Number of topics")
        ax.set_ylabel("Coherence")
        ax.plot(k_values, coherences)#[0.2958, 0.3150, 0.3251, 0.3189, 0.3224, 0.3228, 0.3262, 0.3204]) [0.33, 0.3509, 0.3496, 0.3449, 0.3447, 0.3443, 0.3476, 0.3446, 0.3414])
        ax.annotate('Best number of topics = %d,\n Coherence = %.4f'%(k_values[np.argmax(coherences)], max(coherences)),
                    xy=(k_values[np.argmax(coherences)], max(coherences)), xycoords='data',
                    xytext=(k_values[np.argmax(coherences)], max(coherences) - 0.01), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top')
        plt.tight_layout()
        plt.savefig("coherence_topics_%d.png"%int(datetime.timestamp(datetime.now())))

    def calculate_results(self, big_dataset, i, topic_keywords, compare_to_other_cols = ["plant_products_search", "animal_products_search",\
        "geo_regions", "countries_mentioned", "interventions_found_raw"], to_lower = False):
        key_words = []
        for keyword in big_dataset["keywords"].values[i].split(";"):
            for m in re.finditer("\((.*?)\)", keyword):
                res = m.group(1)
                key_words.append(res.strip().lower())
            keyword = re.sub("\(.*?\)", " ", keyword)
            keyword = " ".join(text_normalizer.get_stemmed_words_inverted_index(keyword))
            key_words.append(keyword.strip().lower() if to_lower else keyword.strip())
        key_words = list(filter(None, key_words))
        if len(key_words) == 0:
            return ()
        ok_keywords_topics = set([w.lower() for w in key_words]).intersection(set([w.lower() for w in topic_keywords]))
        ok_keywords_author = set([w.lower() for w in key_words]).intersection(set([w.lower() for w in topic_keywords]))
        for key_word in topic_keywords:
            key_word_embedding = self.get_mean_vector(key_word)
            for key in key_words:
                if utils.normalized_levenshtein_score(
                        key_word,key) >= 0.77 or (1 - spatial.distance.cosine(key_word_embedding, self.get_mean_vector(key))) >= 0.35:
                    ok_keywords_topics.add(key_word.lower())
                    ok_keywords_author.add(key.lower())
                    #print(key_word, " $ ", key)
        additional_keywords = set()
        if len(compare_to_other_cols) > 0:
            all_other_keywords = set()
            for column in compare_to_other_cols:
                all_other_keywords = all_other_keywords.union(set(big_dataset[column].values[i]))

            for key_word in all_other_keywords:
                key_word_embedding = self.get_mean_vector(key_word)
                for key in key_words:
                    if utils.normalized_levenshtein_score(
                            key_word,key) >= 0.77 or (1 - spatial.distance.cosine(key_word_embedding, self.get_mean_vector(key))) >= 0.5:
                        additional_keywords.add(key.lower())
        #print(ok_keywords_topics)
        #print(ok_keywords_author)
        #print(topic_keywords)
        #print(key_words)
        #print(additional_keywords)
        return (len(ok_keywords_topics)/len(topic_keywords), len(ok_keywords_author)/len(key_words),\
                len(ok_keywords_author.union(additional_keywords)) / len(key_words))

    def get_info_per_dataset_type(self, results, datasets_science_journals):
        res_info = {}
        for idx, res in enumerate(results):
            if len(res) == 0:
                continue
            name = "Grey literature"
            if big_dataset["dataset"].values[idx] in datasets_science_journals:
                name = "Science journals"
            if "total" not in res_info:
                res_info["total"] = {"Grey literature":0, "Science journals":0}
            res_info["total"][name] += 1
            if "accur" not in res_info:
                res_info["accur"] = {"Grey literature":0, "Science journals":0}
            res_info["accur"][name] += res[0]
            if "recall" not in res_info:
                res_info["recall"] = {"Grey literature":0, "Science journals":0}
            res_info["recall"][name] += res[1]
            if "recall_full" not in res_info:
                res_info["recall_full"] = {"Grey literature":0, "Science journals":0}
            res_info["recall_full"][name] += res[2]
            if ">50" not in res_info:
                res_info[">50"] = {"Grey literature":0, "Science journals":0}
            if res[1] >= 0.5:
                res_info[">50"][name] += 1
            if ">70" not in res_info:
                res_info[">70"] = {"Grey literature":0, "Science journals":0}
            if res[1] >= 0.7:
                res_info[">70"][name] += 1
            if "zero" not in res_info:
                res_info["zero"] = {"Grey literature":0, "Science journals":0}
            if res[1] == 0 or res[0] == 0:
                res_info["zero"][name] += 1
        for column in ["accur", "recall", "recall_full"]:
            for dataset in ["Grey literature", "Science journals"]:
                res_info[column][dataset] /= res_info["total"][dataset]
        return res_info

    def calculate_docs_with_threshold_intersection(self, results, threshold):
        cnt_thr = 0
        for res in results:
            if len(res) == 0:
                continue
            if res[0] != 0 and res[1] >= threshold:
                cnt_thr += 1
        return cnt_thr

    def calculate_full_info(self, big_dataset):
        results = []
        t = time()
        for i in range(len(big_dataset)):
            if i %5000 == 0:
                print(i)
                print(time() - t)
                t = time()
            topic_key_words = big_dataset["topics_keywords"].values[i]
            res = self.calculate_results(big_dataset, i, topic_key_words)
            if len(res) > 0 and res[1] == 0:
                res = self.calculate_results(big_dataset, i, topic_key_words, to_lower = True)
            results.append(res)
        cnt_zero = 0
        for idx,res in enumerate(results):
            if len(res) == 0:
                continue
            if res[2] == 0:
                cnt_zero += 1
        print("Count of zero intersection documents ", cnt_zero)
        print("Count of 50\% intersection documents ",self.calculate_docs_with_threshold_intersection(results, 0.5))
        print("Count of 70\% intersection documents ",self.calculate_docs_with_threshold_intersection(results, 0.7))
        accur, recall, recall_full = 0,0,0
        cnt_all = 0
        for res in results:
            if len(res) == 0:
                continue
            cnt_all += 1
            accur += res[0]
            recall += res[1]
            recall_full += res[2]
        if cnt_all > 0:
            print("Accuracy: {}, Recall: {}, Recall full: {}".format(accur/ cnt_all, recall / cnt_all, recall_full / cnt_all))
        return results