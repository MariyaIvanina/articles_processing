from text_processing import text_normalizer
from gensim.models.phrases import Phrases, Phraser
import os
import gensim
from gensim.models import FastText
import numpy as np

class SynonymsProcessor:

	def __init__(self, folder="../model"):
		self.load_models(folder)

	def load_models(self, folder="../model"):
		self.phrases = Phraser.load(os.path.join(folder, "phrases_bigram.model"))
		self.phrases_3gram = Phraser.load(os.path.join(folder, "phrases_3gram.model"))
		self.google_model = gensim.models.Word2Vec.load(os.path.join(folder, "google_plus_our_dataset/", "google_plus_our_dataset.model"))
		self.google_2_and_3_bigrams_model = gensim.models.Word2Vec.load(os.path.join(folder, "google_2_and_3_bigrams_our_dataset/", "google_2_and_3_bigrams_our_dataset.model"))
		self.fast_text_model = FastText.load(os.path.join(folder, "fast_text_our_dataset/", "fast_text_our_dataset.model"))

	def normalize_query(self, query):
		return " ".join(text_normalizer.get_stemmed_words_inverted_index(text_normalizer.normalize_text(query)))

	def find_synonyms(self, model, query, n_words):
		words = []
		try:
			for word in model.wv.most_similar(self.normalize_query(query), topn=n_words):
				normalized_word = word[0]
				words.append(normalized_word)
		except:
			print("Word %s is not found in the vocabulary" %query)
		try:
			for word in model.wv.similar_by_vector(self.get_average_embedding(self.normalize_query(query)), topn=n_words+1):
				normalized_word = word[0]
				if normalized_word == self.normalize_query(query):
					continue
				if word[1] < 0.05:
					continue
				words.append(normalized_word)
		except:
			print("Similar words for word %s were not found" %query)
		return words[:n_words]

	def print_synonyms(self, model, query, n_words):
		synonyms = self.find_synonyms(model, query, n_words)
		for word in synonyms:
			print(word)

	def get_synonyms(self, word):
		return self.find_synonyms(self.google_2_and_3_bigrams_model, word, 20)
	
	def get_word_expression_embedding(self, model, sentence, use_3_gram=False):
		word_expressions_cnt = 0
		phrases = self.phrases[text_normalizer.get_stemmed_words_inverted_index(text_normalizer.normalize_text(sentence.strip()))]
		vec = np.zeros(300)
		if use_3_gram:
			phrases = self.phrases_3gram[phrases]
		for phr in phrases:
			word = phr.replace("_", " ")
			if word in model.wv:
				vec += model.wv[word]
				word_expressions_cnt += 1
			else:
				normalized_word = " ".join(text_normalizer.get_stemmed_words_inverted_index(word.lower()))
				if normalized_word in model.wv:
					vec += model.wv[normalized_word]
					word_expressions_cnt += 1
		return vec if word_expressions_cnt == 0 else vec/word_expressions_cnt
	
	def get_average_embedding(self, sentence):
		return self.get_word_expression_embedding(
			self.google_model, sentence, use_3_gram=False)