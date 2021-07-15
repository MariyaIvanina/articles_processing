from text_processing import text_normalizer
from gensim.models.phrases import Phrases, Phraser
import os
from gensim.models import FastText
import gensim
import pickle
from time import time

class SynonymsTrainer:

    def __init__(self, folder, columns_to_use_for_sentences=[
            "title", "abstract"], columns_to_use_as_keywords=[
            "keywords", "identificators"]):
        self.columns_to_use_as_keywords = columns_to_use_as_keywords
        self.columns_to_use_for_sentences = columns_to_use_for_sentences
        self.folder = folder

    def save_model(self, model, folder, model_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if model != None:
            model.save(os.path.join(folder, model_name))

    def train_phrases(self, articles_df):
        print("Started training phrases")
        t0 = time()
        if os.path.exists(os.path.join(self.folder, "phrases_bigram.model")) and os.path.exists(os.path.join(self.folder, "phrases_3gram.model")):
            self.phrases = Phraser.load(os.path.join(self.folder, "phrases_bigram.model"))
            self.phrases_3gram = Phraser.load(os.path.join(self.folder, "phrases_3gram.model"))
        else:
            phrase_sentences = []
            for i in range(len(articles_df)):
                if i % 5000 == 0 or i == len(articles_df) - 1:
                    print("Processed %d articles" % i)
                for column in self.columns_to_use_for_sentences:
                    text = ""
                    if column.lower() == "title":
                        text = articles_df["title"].values[i].lower() if articles_df["title"].values[i].isupper() else articles_df["title"].values[i]
                    text = articles_df[column].values[i]
                    text = text_normalizer.remove_accented_chars(text)
                    phrase_sentences.extend(
                        text_normalizer.normalize_sentences(
                            text_normalizer.tokenize_words(text)))
                keywords = ""
                for column in self.columns_to_use_as_keywords:
                    keywords = keywords + articles_df[column].values[i] + " ; "
                keywords = text_normalizer.remove_accented_chars(keywords)

                if keywords.strip() != "":
                    phrase_sentences.extend(
                        text_normalizer.normalize_sentences(
                            text_normalizer.tokenize_words(keywords)))

            self.phrases = Phraser(Phrases(phrase_sentences))
            self.phrases_3gram = Phraser(Phrases(self.phrases[phrase_sentences]))
            self.save_model(self.phrases, self.folder, "phrases_bigram.model")
            self.save_model(self.phrases_3gram, self.folder, "phrases_3gram.model")
        print("Finished training phrases: %0.2f s"%(time() - t0))

    def load_models(self, folder="../model"):
        self.phrases = Phraser.load(os.path.join(folder, "phrases_bigram.model"))
        self.phrases_3gram = Phraser.load(os.path.join(folder, "phrases_3gram.model"))
        self.google_model = gensim.models.Word2Vec.load(os.path.join(folder, "google_plus_our_dataset/", "google_plus_our_dataset.model"))
        self.google_2_and_3_bigrams_model = gensim.models.Word2Vec.load(os.path.join(folder, "google_2_and_3_bigrams_our_dataset/", "google_2_and_3_bigrams_our_dataset.model"))
        self.fast_text_model = FastText.load(os.path.join(folder, "fast_text_our_dataset/", "fast_text_our_dataset.model"))

    def save_models(self, folder ="../model"):
        self.save_model(self.phrases, folder, "phrases_bigram.model")
        self.save_model(self.phrases_3gram, folder, "phrases_3gram.model")
        self.save_model(self.google_2_and_3_bigrams_model, os.path.join(folder, "google_2_and_3_bigrams_our_dataset/"), "google_2_and_3_bigrams_our_dataset.model")
        self.save_model(self.google_model, os.path.join(folder, "google_plus_our_dataset/"), "google_plus_our_dataset.model")
        self.save_model(self.fast_text_model, os.path.join(folder, "fast_text_our_dataset/"), "fast_text_our_dataset.model")

    def get_phrased_sentences(self, articles_df, use_3_grams):
        print("Started processing train corpus")
        t0 = time()
        train_sentences = []
        for i in range(len(articles_df)):
            if i % 5000 == 0 or i == len(articles_df) - 1:
                print("Processed %d articles" % i)
            sentence = []
            for column in self.columns_to_use_for_sentences:
                text = ""
                if column.lower() == "title":
                    text = articles_df["title"].values[i].lower() if articles_df["title"].values[i].isupper() else articles_df["title"].values[i]
                text = articles_df[column].values[i]
                text = text_normalizer.remove_accented_chars(text)
                sentence.extend(
                    text_normalizer.get_phrased_sentence(text,
                    self.phrases, self.phrases_3gram if use_3_grams else None))
            train_sentences.append(sentence)
        print("Finished processing train corpus: %0.2f s"%(time() - t0))
        return train_sentences

    def filter_abbreviations(self, train_sentences):
        new_sentences = []
        for sent in train_sentences:
            new_sentences.append([word for word in sent if not text_normalizer.is_abbreviation(word)])
        return new_sentences

    def prepare_sentences_for_learning(self, articles_df):
        self.train_sentences = self.get_phrased_sentences(articles_df, False)
        self.train_sentences_without_abbreviations = self.filter_abbreviations(self.train_sentences)
        self.train_sentences_3grams = self.get_phrased_sentences(articles_df, True)

    def train_fast_text_model(self):
        print("Fast text training started...")
        t0 = time()
        if os.path.exists(os.path.join(self.folder, "fast_text_our_dataset/", "fast_text_our_dataset.model")):
            self.fast_text_model = FastText.load(os.path.join(self.folder, "fast_text_our_dataset/", "fast_text_our_dataset.model"))
        else:
            self.fast_text_model = FastText(size=300, window=6, min_count=7)  
            self.fast_text_model.build_vocab(sentences=self.train_sentences_without_abbreviations)
            print("Vocabulary size: ", len(self.fast_text_model.wv.vocab))
            self.fast_text_model.train(self.train_sentences_without_abbreviations,total_examples=len(self.train_sentences_without_abbreviations), epochs=10)
            self.save_model(self.fast_text_model, os.path.join(self.folder, "fast_text_our_dataset/"), "fast_text_our_dataset.model")
        print("Fast text training finished: %0.2f s"%(time() - t0))

    def train_google_model(self):
        print("Google model training started...")
        t0 = time()
        if os.path.exists(os.path.join(self.folder, "google_plus_our_dataset/", "google_plus_our_dataset.model")):
            self.google_model = gensim.models.Word2Vec.load(os.path.join(self.folder, "google_plus_our_dataset/", "google_plus_our_dataset.model"))
        else:
            self.google_model = gensim.models.Word2Vec(size=300, window = 6, min_count = 7)
            self.google_model.build_vocab(self.train_sentences)
            print("Vocabulary size: ", len(self.google_model.wv.vocab))
            self.google_model.intersect_word2vec_format("../model/GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
            self.google_model.train(self.train_sentences,total_examples=len(self.train_sentences), epochs=10)
            self.save_model(self.google_model, os.path.join(self.folder, "google_plus_our_dataset/"), "google_plus_our_dataset.model")
        print("Google model training finished: %0.2f s"%(time() - t0))

    def train_google_2_and_3_grams_model(self):
        training_sentences = self.train_sentences + self.train_sentences_3grams
        print("Google 2 and 3 grams model training started...")
        t0 = time()
        if os.path.exists(os.path.join(self.folder, "google_2_and_3_bigrams_our_dataset/", "google_2_and_3_bigrams_our_dataset.model")):
            self.google_2_and_3_bigrams_model = gensim.models.Word2Vec.load(os.path.join(self.folder, "google_2_and_3_bigrams_our_dataset/", "google_2_and_3_bigrams_our_dataset.model"))
        else:
            self.google_2_and_3_bigrams_model = gensim.models.Word2Vec(size=300, window = 6, min_count = 7)
            self.google_2_and_3_bigrams_model.build_vocab(training_sentences)
            print("Vocabulary size: ", len(self.google_2_and_3_bigrams_model.wv.vocab))
            self.google_2_and_3_bigrams_model.intersect_word2vec_format("../model/GoogleNews-vectors-negative300.bin",binary=True, lockf=1.0)
            self.google_2_and_3_bigrams_model.train(training_sentences,total_examples=len(training_sentences), epochs=10)
            self.save_model(self.google_2_and_3_bigrams_model, os.path.join(self.folder, "google_2_and_3_bigrams_our_dataset/"), "google_2_and_3_bigrams_our_dataset.model")
        print("Google 2 and 3 grams model training finished: %0.2f s"%(time() - t0))

    def train_models(self, all_models=True, google_model=False, google_model_2_3_grams=False, fast_text=False):
        if all_models or google_model:
            self.train_google_model()
        if all_models or google_model_2_3_grams:
            self.train_google_2_and_3_grams_model()
        if all_models or fast_text:
            self.train_fast_text_model()

    def find_inverted_index_for_n_grams_sentences(self):
        invert_index = {}
        for idx,sent in enumerate(self.train_sentences):
            for word in sent:
                if word not in invert_index:
                    invert_index[word] = set()
                invert_index[word].add(idx)
                
        for idx,sent in enumerate(self.train_sentences_3grams):
            for word in sent:
                if word not in invert_index:
                    invert_index[word] = set()
                invert_index[word].add(idx)
        for word in invert_index:
            invert_index[word] = len(invert_index[word])
        return invert_index

    def generate_subexpressions(self, expression):
        words = expression.split()
        generated_words = set()
        for i in range(len(words)):
            word = words[i]
            generated_words.add(word)
            for j in range(i+1, len(words)):
                word = word + " " + words[j]
                generated_words.add(word)
        return generated_words

    def find_popular_expressions_index(self, invert_index):
        popular_expressions = {}
        for word in invert_index:
            if " " in word:
                generated_words = self.generate_subexpressions(word)
                for generated_word in generated_words:
                    if generated_word == word:
                        continue
                    if generated_word not in popular_expressions:
                        popular_expressions[generated_word] = set()
                    popular_expressions[generated_word].add(word)
        for word in popular_expressions:
            sorted_words = []
            for expression in popular_expressions[word]:
                if expression in invert_index:
                    sorted_words.append((expression, invert_index[expression]))
            sorted_words = sorted(sorted_words, key=lambda x:x[1],reverse=True)
            popular_expressions[word] = sorted_words
        return popular_expressions

    def save_phrases_inverted_index_and_popular_expressions(self, folder="../model"):
        invert_index = self.find_inverted_index_for_n_grams_sentences()
        popular_expressions = self.find_popular_expressions_index(invert_index)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder,'invert_index_and_popular_expressions.pckl'), 'wb') as f:
            pickle.dump([invert_index, popular_expressions], f)


