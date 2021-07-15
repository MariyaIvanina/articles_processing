from text_processing import text_normalizer
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from joblib import dump, load
import pickle
import os
import spacy
import numpy as np
from gensim.models.phrases import Phrases, Phraser
import re
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from utilities import excel_writer
import networkx as nx
from scipy import spatial
import gensim
from nltk.tag.perceptron import PerceptronTagger
import math

pos_tagger = PerceptronTagger()
nlp = spacy.load('en_core_web_sm')

class TopicModeler:

    def __init__(self, filter_word_list, n_components, max_df = 0.3, min_df=0.0015, use_standalone_words=False, use_3_grams = False, train = False,
        folder_for_wordvec="../model/synonyms_retrained", use_word2vec_for_keywords=True):
        self.n_components = n_components
        self.filter_word_list = filter_word_list
        self.tf_vectorizer_for_topics = TfidfVectorizer(analyzer=self.filter_words_for_topics, max_df = max_df, min_df = min_df, use_idf=True)
        self.phrases = Phraser.load(os.path.join(folder_for_wordvec, "phrases_bigram.model"))
        self.phrases_3gram = Phraser.load(os.path.join(folder_for_wordvec, "phrases_3gram.model"))
        if use_word2vec_for_keywords and train:
            self.google_model = gensim.models.Word2Vec.load(os.path.join(folder_for_wordvec, "google_plus_our_dataset/google_plus_our_dataset.model"))
        self.use_3_grams = use_3_grams
        self.use_standalone_words = use_standalone_words
        self.use_word2vec_for_keywords = use_word2vec_for_keywords
        self.max_df = max_df
        self.noun_phrases = set()

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        dump(self.nmf_topics, os.path.join(folder,'nmf_model.joblib'))
        pickle.dump(self.tf_vectorizer_for_topics, open(os.path.join(folder,'tfidf.pickle'), "wb"))
        pickle.dump(self.tf_topics, open(os.path.join(folder,'tf_topics.pickle'), "wb"))
        pickle.dump(self.p_topic, open(os.path.join(folder,'p_topic.pickle'), "wb"))
        pickle.dump(self.topic_keywords, open(os.path.join(folder,'topic_keywords.pickle'), "wb"))
        pickle.dump(self.topic_words, open(os.path.join(folder,'topic_words.pickle'), "wb"))
        pickle.dump(self.noun_phrases, open(os.path.join(folder,'noun_phrases.pickle'), "wb"))

    def load_model(self, folder):
        self.tf_vectorizer_for_topics = pickle.load(open(os.path.join(folder,'tfidf.pickle'), 'rb'))
        self.nmf_topics = load(os.path.join(folder,'nmf_model.joblib'))
        self.tf_topics = load(open(os.path.join(folder,'tf_topics.pickle'), "rb"))
        self.p_topic = load(open(os.path.join(folder,'p_topic.pickle'), "rb"))
        if len(self.p_topic) > 0:
            self.n_components = len(self.p_topic[0])
        if os.path.exists(os.path.join(folder,'noun_phrases.pickle')):
            self.noun_phrases = load(open(os.path.join(folder,'noun_phrases.pickle'), "rb"))
        if os.path.exists(os.path.join(folder,'topic_keywords.pickle')):
            self.topic_keywords = load(open(os.path.join(folder,'topic_keywords.pickle'), "rb"))
        if os.path.exists(os.path.join(folder,'topic_words.pickle')):
            self.topic_words = load(open(os.path.join(folder,'topic_words.pickle'), "rb"))

    def contains_filter_words(self, word_expression, filter_word_list):
        for word in word_expression.split():
            if word in filter_word_list:
                return True
        return False

    def contains_words_from_list(self, word_expression, words_to_filter):
        for word in words_to_filter:
            if re.search(r"\b%s\b"%word, word_expression) != None:
                return True
        return False

    def has_noun(self, word):
        for w in pos_tagger.tag(word.split()):
            if "NN" in w[1] or "VBG" in w[1]:
                return True
        return False

    def is_verb(self, word):
        if len(word.split()) > 1:
            return False
        for w in pos_tagger.tag(word.split()):
            if "VB" in w[1] and "VBG" not in w[1]:
                return True
            if "NN" in w[1] or "VBG" in w[1]:
                self.noun_phrases.add(w[0])
        return False

    def filter_words_for_topics(self, words_array):
        words_to_filter = ["paper", "article", "result", "non pr","review","literature","year","data", "study","institute","university","department","date", "sub", "part"]
        filtered_words = []
        for word in words_array:
            word_normalized = word.strip().lower()
            if len(word_normalized) > 1 and word_normalized not in text_normalizer.stopwords_all and \
             word_normalized not in self.filter_word_list and not self.contains_filter_words(word_normalized, self.filter_word_list)\
              and not self.contains_words_from_list(word, words_to_filter)\
              and (not re.search("\d+",word) or text_normalizer.is_abbreviation(word))\
              and not text_normalizer.has_word_with_one_non_digit_symbol(word)\
              and not self.is_verb(word) and text_normalizer.calculate_common_topic_score(word) >= 0.2:
                filtered_words.append(word)
        return filtered_words

    def create_text_for_article(self, articles_df, index):
        title = text_normalizer.remove_accented_chars(articles_df["title"].values[index].lower()\
            if articles_df["title"].values[index].isupper() else articles_df["title"].values[index])
        abstract = text_normalizer.remove_accented_chars(articles_df["abstract"].values[index])
        keywords = text_normalizer.remove_accented_chars(articles_df["keywords"].values[index] + " ; " + articles_df["identificators"].values[index])
        sentence = []
        if self.use_standalone_words:
            sentence.extend(text_normalizer.get_phrased_sentence(title, None, None))
            sentence.extend(text_normalizer.get_phrased_sentence(abstract, None, None))
            sentence.extend(text_normalizer.get_phrased_sentence(keywords.lower(), None, None))
        sentence.extend(text_normalizer.get_phrased_sentence(title, self.phrases, self.phrases_3gram if self.use_3_grams else None))
        sentence.extend(text_normalizer.get_phrased_sentence(abstract, self.phrases, self.phrases_3gram if self.use_3_grams else None))
        sentence.extend(text_normalizer.get_phrased_sentence(keywords.lower(), self.phrases, self.phrases_3gram if self.use_3_grams else None))
        return sentence

    def get_text_to_train(self, articles_df):
        train_sentences_for_topics = []
        for i in range(len(articles_df)):
            train_sentences_for_topics.append(self.create_text_for_article(articles_df, i))
        return train_sentences_for_topics

    def change_weight_tf_idf(self):
        for word in self.tf_vectorizer_for_topics.vocabulary_:
            if len(word.split()) >= 2 and word in self.noun_phrases:
                self.tf_topics[:, self.tf_vectorizer_for_topics.vocabulary_[word]] *= 4
            if word not in self.noun_phrases:
                self.tf_topics[:, self.tf_vectorizer_for_topics.vocabulary_[word]] /= 2
            if text_normalizer.is_abbreviation(word):
                self.tf_topics[:, self.tf_vectorizer_for_topics.vocabulary_[word]] *= 2
            if text_normalizer.contain_full_name(word, ["small scale", "small holder", "smallholder"]):
                self.tf_topics[:, self.tf_vectorizer_for_topics.vocabulary_[word]] /= 2

    def model_topics(self,articles_df, n_top_words = 10):
        print("Extracting tf features for NMF...")
        t0 = time()
        self.train_sentences_for_topics = self.get_text_to_train(articles_df)
        print("Processed texts %0.3fs."%(time() - t0))
        t1 = time()
        self.tf_topics = self.tf_vectorizer_for_topics.fit_transform(self.train_sentences_for_topics)
        print("Performed tf-idf vectorizing %0.3fs."%(time() - t1))
        self.initialize_dictionary_about_nouns()
        self.change_weight_tf_idf()
        print("done in %0.3fs." % (time() - t0))

        print("Fitting NMF models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (len(self.train_sentences_for_topics), self.tf_topics.shape[1]))
        self.nmf_topics = NMF(n_components=self.n_components,
                                        random_state=0)
        t0 = time()
        self.p_topic = self.nmf_topics.fit_transform(self.tf_topics)
        print("NMF finished")
        if self.use_word2vec_for_keywords:
            self.topic_keywords = self.find_topic_keywords()
            self.topic_words = self.find_topic_words()
        print("done in %0.3fs." % (time() - t0))

    def get_mean_vector(self, key_word):
        key_word_embedding = np.zeros(300)
        all_words_cnt = 0
        for phr in self.phrases[text_normalizer.get_stemmed_words_inverted_index(key_word)]:
            if phr.replace("_"," ") in self.google_model.wv:
                key_word_embedding += self.google_model.wv[phr.replace("_"," ")]
                all_words_cnt += 1
        key_word_embedding /= all_words_cnt
        return key_word_embedding

    def get_top_words(self, n_top_words):
        tf_feature_names_topics = self.tf_vectorizer_for_topics.get_feature_names()
        all_topics = []
        for i in range(self.n_components):
            all_topics.append([tf_feature_names_topics[w] for w  in self.nmf_topics.components_[i].argsort()[::-1]][:n_top_words])
        return all_topics

    def find_topic_keywords(self, search_n_top_words = 100, n_top_words = 20):
        topics = self.get_noun_phrases(search_n_top_words = search_n_top_words, all_noun_words = False)
        all_topics = self.get_top_words(3)
        topic_keywords = []
        for ind in range(self.n_components):
            new_res = []
            for w in topics[ind]:
                if text_normalizer.calculate_common_topic_score(w) < 0.2:
                    continue
                tmp_val = 0
                cnt_val = 0
                for w1 in all_topics[ind]:
                    partial_res = (1 - spatial.distance.cosine(self.get_mean_vector(w), self.get_mean_vector(w1)))
                    if partial_res == partial_res:
                        tmp_val += partial_res
                        cnt_val += 1
                simil = tmp_val / cnt_val if cnt_val > 0 else 0
                new_res.append((w, simil))
            new_res = [w[0] for idx, w in enumerate(sorted(new_res, key = lambda x:x[1], reverse = True)) if idx < n_top_words or w[1] >= 0.5]
            topic_keywords.append(new_res)
        return topic_keywords

    def initialize_dictionary_about_nouns(self):
        tf_feature_names_topics = self.tf_vectorizer_for_topics.get_feature_names()
        for word in tf_feature_names_topics:
            if word not in self.noun_phrases and self.has_noun(word):
                self.noun_phrases.add(word)

    def get_noun_phrases(self, search_n_top_words = 50, all_noun_words = True):
        tf_feature_names_topics = self.tf_vectorizer_for_topics.get_feature_names()
        topics = []
        for topic_idx, topic in enumerate(self.nmf_topics.components_):
            topic_list = [tf_feature_names_topics[w] for w  in topic.argsort()[::-1]]
            topic_list = [word for word in topic_list if (word in self.noun_phrases and (all_noun_words or " " in word)) or text_normalizer.is_abbreviation(word)]
            topics.append(topic_list[:search_n_top_words])
        return topics

    def find_topic_words(self, n_top_words = 10, search_n_top_words = 50):
        topic_words = self.get_noun_phrases(search_n_top_words = search_n_top_words, all_noun_words = True)
        key_words = []
        for i in range(len(topic_words)):
            words = topic_words[i]
            G = nx.Graph()
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    w = words[i]
                    w1 = words[j]
                    if w != w1:
                        we = (1 - spatial.distance.cosine(self.get_mean_vector(w), self.get_mean_vector(w1)))
                        G.add_edge(w1, w, weight=we)
            try:
                res = nx.pagerank(G, alpha=0.9)
                res_sorted = sorted(res.items(), key= lambda x: x[1], reverse=True)
                res_sorted = [w[0] for w in res_sorted]
            except:
                res_sorted = words
            main_w = [res_sorted[0]]
            for j in range(1, len(res_sorted)):
                w = res_sorted[0]
                w1 = res_sorted[j]
                if w!=w1 and (1 - spatial.distance.cosine(self.get_mean_vector(w), self.get_mean_vector(w1))) >=0.35:
                    main_w.append(w1)
                if len(main_w) == 3:
                    break
            
            key_w = []
            for word in words:
                cnt = 0
                res_sum = 0
                for w in main_w:
                    if w != word and word not in main_w:
                        partial_res = (1 - spatial.distance.cosine(self.get_mean_vector(w), self.get_mean_vector(word)))
                        if partial_res == partial_res:
                            res_sum += partial_res
                            cnt += 1
                if cnt > 0:
                    key_w.append((word, res_sum/cnt))
            key_words.append(main_w + [w[0] for idx, w in enumerate(sorted(key_w, key = lambda x:x[1], reverse = True)) if idx < n_top_words ])
        return key_words

    def find_topic_matrix_for_documents(self, articles_df):
        train_sentences_for_topics = self.get_text_to_train(articles_df)
        tf_topics = self.tf_vectorizer_for_topics.transform(train_sentences_for_topics)
        p_topic = self.nmf_topics.transform(tf_topics)
        p_topic[np.isnan(p_topic)] = 0
        return p_topic

    def print_topics(self, with_topic_number = True, n_top_words=10):
        print("\nTopics in NMF model:")
        for idx, topic in enumerate(self.topic_words):
            if with_topic_number:
                print("Topic #%d:" % (idx+1) + " " + ",".join(topic[:n_top_words]))
            else:
                print(",".join(topic))

    def calculate_statistics_by_topics(self, search_engine_inverted_index, data_df):
        print("Started texts transformation")
        t0 = time()
        p_topic = self.find_topic_matrix_for_documents(data_df)
        print("Finished texts transformation in %0.3fs."%(time() - t0))
        self.doc_topic_repres = np.zeros(self.n_components, dtype = int)
        self.doc_topic_max = np.zeros(self.n_components)
        self.doc_topic_doc_quantity = np.zeros(self.n_components, dtype = int)
        for i in range(len(data_df)):
            ind = p_topic[i].argsort()[:][::-1] 
            for j in range(len(ind)):
                if p_topic[i][ind[j]] > self.doc_topic_max[ind[j]]:
                    self.doc_topic_max[ind[j]] = p_topic[i][ind[j]]
                    self.doc_topic_repres[ind[j]] = i
        self.chosen_topics = []
        for i in range(len(data_df)):
            ind = p_topic[i].argsort()[:][::-1] 
            topics_chosen = []
            for j in range(len(ind[:3])):
                if ind[j] not in topics_chosen:
                    self.doc_topic_doc_quantity[ind[j]] += 1
                    topics_chosen.append(ind[j])
            for j in range(len(ind)):
                if p_topic[i][ind[j]] / self.doc_topic_max[ind[j]] >= 0.3 and ind[j] not in topics_chosen:
                    self.doc_topic_doc_quantity[ind[j]] += 1
                    topics_chosen.append(ind[j])
            self.chosen_topics.append(topics_chosen)
        self.calculate_topics_statistics_by_keywords(search_engine_inverted_index, data_df)

    def calculate_topics_statistics_by_keywords(self, search_engine_inverted_index, articles_df):
        print("Started topics keywords finding")
        self.doc_topic_quantity_by_index = np.zeros(self.n_components,dtype=int)
        for idx, topic in enumerate(self.topic_keywords):
            doc_set = set()
            for word in topic:
                res = search_engine_inverted_index.find_articles_with_keywords([word], 0.95)
                doc_set = doc_set.union(res)
            self.doc_topic_quantity_by_index[idx] = len(doc_set - set([i for i in range(len(self.chosen_topics)) if idx in self.chosen_topics[i]]))
        print("Finished topics keywords finding")

    def fill_topic_for_articles(self, articles_df, search_engine_inverted_index, column_name="topics", topic_mode="raw topic"):
        articles_df[column_name] = ""
        for article in range(len(articles_df)):
            articles_df[column_name].values[article] = set()
            for topic_num in self.chosen_topics[article]:
                if topic_mode == "raw topic":
                    articles_df[column_name].values[article].add(",".join(self.topic_words[topic_num][:3]))
                elif topic_mode == "topic keywords":
                    for ind_word in range(3):
                        articles_df[column_name].values[article].add(self.topic_words[topic_num][ind_word])
                elif topic_mode == "topic hierarchy":
                    articles_df[column_name].values[article].add(self.topic_words[topic_num][0] + "/")
        for idx, topic in enumerate(self.topic_keywords):
            for word in topic:
                for article in search_engine_inverted_index.find_articles_with_keywords([word], 0.95):
                    if topic_mode == "raw topic":
                        articles_df[column_name].values[article].add(",".join(self.topic_words[idx][:3]))
                    elif topic_mode == "topic keywords":
                        articles_df[column_name].values[article].add(word)
                        articles_df[column_name].values[article].add(self.topic_words[idx][0])
                    elif topic_mode == "topic hierarchy":
                        articles_df[column_name].values[article].add(self.topic_words[idx][0] + "/" + word)
                        articles_df[column_name].values[article].add(self.topic_words[idx][0] + "/")
        for i in range(len(articles_df)):
            articles_df[column_name].values[i] = list(articles_df[column_name].values[i])
        return articles_df

    def export_statistics(self,articles_df, filename):
        topic_rows = []
        for i in range(self.n_components):
            topic_rows.append(("Topic %d:"%(i+1), ", ".join(self.topic_words[i][:10]), ", ".join(self.topic_keywords[i][:10]), self.doc_topic_doc_quantity[i] + self.doc_topic_quantity_by_index[i]))
        excel_writer.ExcelWriter().save_data_in_excel(topic_rows, ["Topics Number", "Top 10 topic words", "Top 10 topic keywords", "Articles with the topic"],\
        filename, column_probabilities=["Articles with the topic"], column_width=[("Topics Number",15),("Top 10 topic words", 100),("Top 10 topic keywords", 100)])




