import pandas as pd
import gensim
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score
from gensim.models.phrases import Phrases, Phraser 
from text_processing import text_normalizer
import pickle
from joblib import dump,load
from sklearn.svm import SVC  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from text_processing import topic_modeling
from copy import deepcopy

class InterventionLabeling:

    def __init__(self, google_models_folder = "../model",
            label_non_intervention=6, class_weights={1:2, 2:2, 3:3, 4:5, 5:5, 6:3 },
            binary=False):
        self.binary = binary
        self.google_model = gensim.models.Word2Vec.load(os.path.join(google_models_folder,"google_plus_our_dataset/", "google_plus_our_dataset.model"))
        print("Google word2vec loaded")
        self.fast_text_model = gensim.models.Word2Vec.load(os.path.join(google_models_folder,"fast_text_our_dataset/", "fast_text_our_dataset.model"))
        print("Fast text loaded")
        self.phrases = Phraser.load(os.path.join(google_models_folder,"phrases_3gram.model"))
        print("Phrases loaded")
        self.feature_stats = {}
        self.class_weights = class_weights
        self.label_non_intervention = label_non_intervention
        self.filter_words = ["practice", "approach", "application", "intervention",\
         "input","strategy", "policy", "program", "programme", "initiative","technology",\
          "science", "technique", "innovation"]
        filter_word_list = text_normalizer.build_filter_dictionary(["../data/Filter_Geo_Names.xlsx"])
        self.topic_modeler = topic_modeling.TopicModeler(filter_word_list, 50,  use_3_grams=False, train = True,
                                           folder_for_wordvec=google_models_folder, use_standalone_words=True)

    def load_previous_models(self, folder):
        self.topic_modeler.load_model(os.path.join(folder, "topics"))
        
        with open(os.path.join(folder,'feature_stats.pckl'), 'rb') as f:
            self.feature_stats = pickle.load(f)

        with open(os.path.join(folder,'mean_embed_vectors.pckl'), 'rb') as f:
            self.mean_interventions_vectors, self.mean_narrow_concepts_vectors = pickle.load(f)

        self.svclassifier = load(os.path.join(folder,'svcclassifier.joblib'))
        
        self.neigh = load(os.path.join(folder,'neigh.joblib'))
        self.clf = load(os.path.join(folder,'clf.joblib'))
        self.AdaBoost = load(os.path.join(folder,'AdaBoost.joblib'))

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.topic_modeler.save_model(os.path.join(folder, "topics"))
        dump(self.svclassifier, os.path.join(folder,'svcclassifier.joblib'))
        dump(self.neigh, os.path.join(folder,'neigh.joblib'))
        dump(self.clf, os.path.join(folder,'clf.joblib'))
        dump(self.AdaBoost, os.path.join(folder,'AdaBoost.joblib'))

        with open(os.path.join(folder,'feature_stats.pckl'), 'wb') as f:
            pickle.dump(self.feature_stats, f)

        with open(os.path.join(folder,'mean_embed_vectors.pckl'), 'wb') as f:
            pickle.dump([self.mean_interventions_vectors, self.mean_narrow_concepts_vectors], f)

    def cross_validation(self, clf, x_train,y_train,kfolds=10):
        x_train_array = np.asarray(x_train)
        y_train_array = np.asarray(y_train)
        skfolds = StratifiedKFold(n_splits=kfolds, random_state=42)
        scores = []
        for train_index, test_index in skfolds.split(x_train,y_train):
            clone_clf = clone(clf)
            x_train_folds = x_train_array[train_index]
            y_train_folds = y_train_array[train_index]
            x_test_folds = x_train_array[test_index]
            y_test_folds = y_train_array[test_index]
            
            clone_clf.fit(x_train_folds, y_train_folds)
            y_pred = clone_clf.predict(x_test_folds)
            scores.append(f1_score(y_test_folds, y_pred, average="macro"))
        return np.asarray(scores)

    def get_distance_to_vectors(self, data, mean_interventions_vectors):
        distances = [distance.euclidean(data, mean_vector) for mean_vector in mean_interventions_vectors]
        return distances

    def normalize_distance_vec(self, data, mean_vec):
        vec = self.get_distance_to_vectors(
            self.get_word_expression_embedding(data), mean_vec)
        return self.softmax((vec - np.mean(vec))/np.std(vec)) - 0.5

    def get_word_expression_embedding(self, sentence, filter_apply = False):
        word_expressions = set()
        for sent in sentence.split(";"):
            for phr in self.phrases[text_normalizer.get_stemmed_words_inverted_index(text_normalizer.normalize_text(sent.strip()))]:
                if not filter_apply:
                    word_expressions.add(phr.replace("_", " "))
        vec = np.zeros(300)
        for word in word_expressions:
            if word in self.google_model.wv:
                vec += self.google_model.wv[word]
            elif word in self.fast_text_model.wv:
                vec += self.fast_text_model.wv[word]
            else:
                print("%s is not found (%s)"%(word, sentence))
        return vec if len(word_expressions) == 0 else vec/(len(word_expressions))

    def get_word_expression_embedding_weighted(self, sentence, filter_apply = False):
        word_expressions = set()
        for sent in sentence.split(";"):
            for phr in self.phrases[text_normalizer.get_stemmed_words_inverted_index(text_normalizer.normalize_text(sent.strip()))]:
                if not filter_apply and phr not in self.filter_words:
                    word_expressions.add(phr)
        vec = np.zeros(300)
        sum_idf = 0.0
        for word in word_expressions:
            if word in self.google_model.wv:
                vec += self.google_model.wv[word] * self.get_word_weight(word.replace("_"," "))
                sum_idf += self.get_word_weight(word.replace("_"," "))
            elif word in self.fast_text_model.wv:
                vec += self.fast_text_model.wv[word] * self.get_word_weight(word.replace("_"," "))
                sum_idf += self.get_word_weight(word.replace("_"," "))
            else:
                print("%s is not found (%s)"%(word, sentence))
        return vec if len(word_expressions) == 0 or sum_idf <= 0.00001 else vec/(len(word_expressions)*sum_idf)

    def normalize(self, values, indexes=[]):
        new_values = deepcopy(values)
        needed_to_calculate =list(set(indexes) - set(self.feature_stats.keys()))
        np_values = np.asarray(values[:,needed_to_calculate])
        try:
            np_min, np_max, np_mean = np_values.min(axis=0), np_values.max(axis=0), np_values.mean(axis=0)
            for i in range(len(needed_to_calculate)):
                index = needed_to_calculate[i]
                if index not in self.feature_stats:
                    self.feature_stats[index] = {"min":np_min[i], "max":np_max[i], "mean":np_mean[i]}
            for index in indexes:
                max_value = self.feature_stats[index]["max"]
                min_value = self.feature_stats[index]["min"]
                for i in range(len(new_values)):
                    new_values[i][index] = (new_values[i][index] -  self.feature_stats[index]["mean"]) / (max_value - min_value)
        except:
            pass
        return new_values

    def get_word_weight(self, word, label_class=0):
        if word not in self.tf_idf_for_all_words.vocabulary_:
            return 1.0
        return self.words_[label_class, self.tf_idf_for_all_words.vocabulary_[word]]*self.tf_idf_for_all_words.idf_[self.tf_idf_for_all_words.vocabulary_[word]]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def linear_normalize(self, x):
        res = x / np.sum(x, axis=1)[:, None]
        res[np.isnan(res)] = 0
        return res

    def get_mean_separate_words(self, train_data, labels, filter_apply = False):
        words_set = [[] for i in range(self.label_non_intervention if not self.binary else 2)]
        mean_vector_interventions = np.zeros((self.label_non_intervention  if not self.binary else 2,300))
        for i in range(len(train_data)):
            data = train_data[i]
            for sent in data.split(";"):
                for phr in self.phrases[text_normalizer.get_stemmed_words_inverted_index(text_normalizer.normalize_text(sent.strip()))]:
                    if phr not in words_set[labels[i]-1]:
                        word_embedding = self.get_word_expression_embedding(phr.replace("_"," "), filter_apply)
                        words_set[labels[i]-1].append(phr.replace("_"," "))
                        mean_vector_interventions[labels[i]- 1] += word_embedding
        for i in range(self.label_non_intervention if not self.binary else 2):
            mean_vector_interventions[i] /= len(words_set[i])
        return mean_vector_interventions

    def get_normalized_distances(self, vec):
        return (vec - np.mean(vec))/(np.std(vec)+10e-6)

    def train_svm(self, train_x, train_y, test_x, test_y):
        svclassifier = SVC(kernel='rbf', probability=True, class_weight=self.class_weights, random_state=41)  
        svclassifier.fit(train_x, train_y)
        
        scores = self.cross_validation(svclassifier, train_x, train_y)
        print(scores)
        print("Cross validation score(F1): %f"%(scores.mean()))
        print("Test accuracy : %f" %(svclassifier.score(test_x,test_y)))
        return svclassifier

    def train_knn(self, train_x_knn, train_y_knn, test_x_knn, test_y_knn, neighbors = 5):
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        neigh.fit(train_x_knn, train_y_knn)
        print("Test accuarcy :%f"%(neigh.score(test_x_knn, test_y_knn)))

        scores = self.cross_validation(neigh, train_x_knn, train_y_knn)
        print(scores)
        print("Cross validation score(F1): %f"%(scores.mean()))
        return neigh

    def train_adaboost(self, ada_boost_x, train_y, ada_boost_test_x, test_y):
        AdaBoost = GradientBoostingClassifier(n_estimators  = 50, max_depth=1, random_state=41)
        AdaBoost.fit(ada_boost_x, train_y)
        print(AdaBoost.score(ada_boost_test_x, test_y))
        print(AdaBoost.score(ada_boost_x, train_y))

        scores = self.cross_validation(AdaBoost, ada_boost_x, train_y)
        print(scores)
        print("Cross validation score(F1): %f"%(scores.mean()))
        return AdaBoost

    def prepare_data_for_topic_modeling(self, train_data):
        part_df = pd.DataFrame(train_data[:, :2], columns=["title", "abstract"])
        part_df["keywords"] = ""
        part_df["identificators"] = ""
        return part_df

    def train_model(self, train_data, test_data):

        train_data = self.normalize(train_data)
        test_data = self.normalize(test_data)

        print("SVM model whole embeddings")
        train_x, train_y = [np.concatenate((self.get_word_expression_embedding(data[0]), self.get_word_expression_embedding(data[0]+";"+data[1]))) for data in train_data], [data[-1] for data in train_data]
        test_x, test_y = [np.concatenate((self.get_word_expression_embedding(data[0]), self.get_word_expression_embedding(data[0]+";"+data[1]))) for data in test_data], [data[-1] for data in test_data]
        self.svclassifier = self.train_svm(train_x, train_y, test_x, test_y)
        y_pred = self.svclassifier.predict(test_x)
        print(confusion_matrix(test_y,y_pred))  
        print(classification_report(test_y,y_pred))
        
        print("SVM model mean embeddings")
        self.mean_interventions_vectors = self.get_mean_separate_words([data[0]+";"+data[1] for data in train_data], [data[-1] for data in train_data])
        self.mean_narrow_concepts_vectors = self.get_mean_separate_words([data[0] for data in train_data], [data[-1] for data in train_data])
        train_x_mean, train_y_mean = [np.concatenate((self.normalize_distance_vec(data[0], self.mean_narrow_concepts_vectors), self.normalize_distance_vec(data[0]+";"+data[1], self.mean_interventions_vectors), self.get_word_expression_embedding(data[0]))) for data in train_data], [data[-1] for data in train_data]
        test_x_mean, test_y_mean = [np.concatenate((self.normalize_distance_vec(data[0], self.mean_narrow_concepts_vectors), self.normalize_distance_vec(data[0]+";"+data[1], self.mean_interventions_vectors), self.get_word_expression_embedding(data[0]))) for data in test_data], [data[-1] for data in test_data]

        self.clf = self.train_svm(train_x_mean,train_y_mean, test_x_mean,test_y_mean)
        print(self.clf.score(train_x_mean, train_y_mean))
        y_pred = self.clf.predict(test_x_mean)
        print(confusion_matrix(test_y_mean,y_pred))  
        print(confusion_matrix(train_y_mean,self.clf.predict(train_x_mean)))
        

        print("KNN model hyponym embeddings")
        train_x_knn, train_y_knn = [self.get_word_expression_embedding(data[0]) for data in train_data], [data[-1] for data in train_data]
        test_x_knn, test_y_knn = [self.get_word_expression_embedding(data[0]) for data in test_data], [data[-1] for data in test_data]

        self.neigh = self.train_knn(train_x_knn, train_y_knn, test_x_knn, test_y_knn)
        y_pred = self.neigh.predict(test_x_knn)
        print(confusion_matrix(test_y_knn,y_pred))

        part_df = self.prepare_data_for_topic_modeling(train_data)
        test_part_df = self.prepare_data_for_topic_modeling(test_data)
        self.topic_modeler.model_topics(part_df)   

        print("AdaBoost model SVM, NN, SVM mean, topics")
        ada_boost_x = np.concatenate((self.svclassifier.predict_proba(train_x), self.neigh.predict_proba(train_x_knn), self.clf.predict_proba(train_x_mean),
            self.linear_normalize(self.topic_modeler.find_topic_matrix_for_documents(part_df))),axis = 1)
        ada_boost_test_x = np.concatenate((self.svclassifier.predict_proba(test_x), self.neigh.predict_proba(test_x_knn), self.clf.predict_proba(test_x_mean),
            self.linear_normalize(self.topic_modeler.find_topic_matrix_for_documents(test_part_df))),axis = 1)

        self.AdaBoost = self.train_adaboost(ada_boost_x,train_y, ada_boost_test_x,test_y)
        y_pred_ad = self.AdaBoost.predict(ada_boost_test_x)
        print(confusion_matrix(test_y,y_pred_ad))
        print(classification_report(test_y,y_pred_ad))

    def predict_class(self, values, return_probs=False):
        ada_boost_data = self.get_data_before_predict(values)
        if return_probs:
            return self.AdaBoost.predict(ada_boost_data), self.AdaBoost.predict_proba(ada_boost_data)
        return self.AdaBoost.predict(ada_boost_data)

    def get_data_before_predict(self, values):
        values = self.normalize(values)
        svm_data = [np.concatenate((self.get_word_expression_embedding(data[0]), self.get_word_expression_embedding(data[0]+";"+data[1]))) for data in values]
        svm_mean_data = [np.concatenate((self.normalize_distance_vec(data[0], self.mean_narrow_concepts_vectors), self.normalize_distance_vec(data[0]+";"+data[1], self.mean_interventions_vectors), self.get_word_expression_embedding(data[0]))) for data in values]
        knn_data = [self.get_word_expression_embedding(data[0]) for data in values]
        part_df = self.prepare_data_for_topic_modeling(values)
        ada_boost_data = np.concatenate((self.svclassifier.predict_proba(svm_data), self.neigh.predict_proba(knn_data), self.clf.predict_proba(svm_mean_data),
            self.linear_normalize(self.topic_modeler.find_topic_matrix_for_documents(part_df))),axis = 1)
        return ada_boost_data
