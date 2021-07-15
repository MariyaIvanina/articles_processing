from text_processing import search_engine_insensitive_to_spelling
from text_processing import text_normalizer
from time import time

class ConceptsMerger:

    def __init__(self, symbols_count = 5):
        self.symbols_count = symbols_count
        self.new_mapping = {}
        self.inverted_dictionary = {}
        self.search_engine_by_letters = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling(symbols_count = self.symbols_count)

    def add_item_to_dict(self, word, docId):
        self.search_engine_by_letters.add_item_to_dict(word, docId)

    def get_weight(self, word, search_engine_inverted_index):
        return len(search_engine_inverted_index.find_articles_with_keywords([word],1.0, extend_with_abbreviations=False))

    def get_frequent_word(self, word, search_engine_inverted_index):
        real_word_count = len(search_engine_inverted_index.find_articles_with_keywords([word], 1.0, extend_with_abbreviations=False))
        without_space_word_count = len(search_engine_inverted_index.find_articles_with_keywords([word.replace(" ", "")], 1.0, extend_with_abbreviations=False))
        if without_space_word_count == 0:
            return word
        if real_word_count == 0 or without_space_word_count/real_word_count > 2:
            return word.replace(" ", "")
        return word
    
    def merge_concepts(self, search_engine_inverted_index, threshold = 0.92):
        start_time = time()
        cnt_groups = 0 
        for first_letters in self.search_engine_by_letters.dictionary_by_first_letters:
            if len(first_letters) <= 1 or (len(first_letters) == 2 and not text_normalizer.is_abbreviation(first_letters)):
                continue
            if (len(first_letters) == 2 and text_normalizer.is_abbreviation(first_letters)) or (len(first_letters) > 2 and len(first_letters) <self.symbols_count):
                self.new_mapping[first_letters] = first_letters
                continue
            cnt_groups += 1
            list_of_words = [
                (word, self.get_weight(word, search_engine_inverted_index)) for word in self.search_engine_by_letters.dictionary_by_first_letters[first_letters]]
            list_of_words = sorted(list_of_words,key = lambda x: x[1],reverse=True)
            for i in range(len(list_of_words)):
                for j in range(i+1, len(list_of_words)):
                    if text_normalizer.are_words_similar(list_of_words[i][0], list_of_words[j][0], threshold):
                        self.new_mapping[list_of_words[j][0]] = list_of_words[i][0] if list_of_words[i][0] not in self.new_mapping else self.new_mapping[list_of_words[i][0]]
                if list_of_words[i][0] not in self.new_mapping:
                    self.new_mapping[list_of_words[i][0]] = list_of_words[i][0]
        print("merge concepts: %d s"%(time()-start_time))
        print("groups to merge: %d"%cnt_groups)
        self.create_inverted_dictionary()
    
    def create_inverted_dictionary(self):
        for key, val in self.new_mapping.items():
            if val not in self.inverted_dictionary:
                self.inverted_dictionary[val] = set()
            self.inverted_dictionary[val].add(key)