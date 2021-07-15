import pickle
from interventions_labeling_lib import hearst_pattern_finder
from text_processing import text_normalizer
import os
import re
import textdistance
from collections import deque

class AbbreviationsResolver:

    def __init__(self, filter_words = []):
        self.abbreviations_finder_dict = {}
        self.filter_words = filter_words

    def save_model(self, folder="..\\model"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump([self.abbreviations_finder_dict,self.resolved_abbreviations, self.words_to_abbreviations], open(os.path.join(folder,"hyponyms_found_abbreviations.pickle"),"wb"))
        self.save_dictionaries(folder)

    def load_model(self, folder="..\\model"):
        self.abbreviations_finder_dict,self.resolved_abbreviations, self.words_to_abbreviations = pickle.load(open(os.path.join(folder,"hyponyms_found_abbreviations.pickle"),"rb"))
        self.sorted_resolved_abbreviations, self.sorted_words_to_abbreviations = pickle.load(open(os.path.join(folder,"resolved_abbreviations.pickle"),"rb"))

    def extract_hyponym_abbreviations(self, articles_df, continue_extract = False, use_prefix="", columns_to_use=["title", "abstract"]):
        if not continue_extract:
            self.abbreviations_finder_dict = {}
        hyponyms_finder_abbreviations = hearst_pattern_finder.HearstPatterns(True,True)
        for i in range(len(articles_df)):
            art_id = articles_df["id"].values[i] if "id" in articles_df.columns else i
            if use_prefix:
                art_id = use_prefix + "_" + str(art_id)
            if art_id in self.abbreviations_finder_dict:
                continue
            if i % 5000 == 0 or i== len(articles_df) -1:
                print("Processed %d articles"%i)
            text = ""
            for column in columns_to_use:
                text = text + articles_df[column].values[i] + " . "
            self.abbreviations_finder_dict[art_id] = hyponyms_finder_abbreviations.find_hyponyms(text)

    def find_abbreviations_in_hyponyms(self):
        self.abbreviations = {}
        for i in self.abbreviations_finder_dict:
            for pat in self.abbreviations_finder_dict[i]:
                abbreviation = ""
                if text_normalizer.is_abbreviation(pat[1]):
                    abbreviation = pat[1]
                if text_normalizer.is_abbreviation(pat[0]):
                    abbreviation = pat[0]
                if abbreviation == "" or len(abbreviation) == 1 or len(abbreviation) > 10:
                    continue
                if abbreviation not in self.abbreviations:
                    self.abbreviations[abbreviation] = []
                self.abbreviations[abbreviation].append(pat)

    def generate_abbreviations(self, words_text):
        generated_words = set()
        if len(words_text) == 0:
            return set()
        if len(words_text) == 1:
            return set([words_text[0][:1].lower(), words_text[0][:2].lower()]) if not text_normalizer.is_abbreviation(words_text[0]) else words_text[0].lower()
        words = self.generate_abbreviations(words_text[1:])
        for word in words:
            generated_words.add(word + "\w?" + (words_text[0].lower() if text_normalizer.is_abbreviation(words_text[0]) else words_text[0][:1].lower()))
            generated_words.add(word + "\w?" + (words_text[0].lower() if text_normalizer.is_abbreviation(words_text[0]) else words_text[0][:2].lower()))
        return generated_words

    def are_words_forming_abbreviations(self, words_text, word, words_count_to_check, word_abbr):
        words_text_pruned = words_text[:min(words_count_to_check, len(words_text))]
        if len(words_text_pruned) < 2:
            return ""
        for subset_abbr in self.generate_abbreviations(words_text_pruned):
            if (re.match(subset_abbr, word.lower()) != None and re.match(subset_abbr, word.lower()).group(0) == word.lower())\
                    or re.match(word_abbr, subset_abbr.replace("\w?","")) != None:
                return " ".join(reversed(words_text_pruned))
        return ""
        
    def find_text_for_abbreviation(self, word, text):
        word = word.replace(" ","")
        words_text = text.split()[::-1]
        if len(words_text) == 1:
            return ""
        word_abbr = "".join([letter+"\w?" for letter in word.lower()])
        words_count_to_check = [len(word), len(word)+1, len(word)+2] + list(range(len(word)-1, 1, -1))
        for length_abbr in words_count_to_check:
            for offset in [0,1,2]:
                words_for_abbreviation = self.are_words_forming_abbreviations(words_text[offset:], word, length_abbr, word_abbr)
                if words_for_abbreviation != "":
                    return words_for_abbreviation
        return ""

    def clean_abbreviation_expression(self, word, text):
        if re.search("%s"%word, text) != None or re.search("%s"%word.lower(), text) != None:
            return ""
        words_text = text.split()
        if len(words_text) <= 2:
            return text
        for last_word_ind in range(len(words_text)-1,2,-1):
            if words_text[last_word_ind] in self.filter_words and self.find_text_for_abbreviation(word, " ".join(words_text[:last_word_ind])) != 0:
                return " ".join(words_text[:last_word_ind])
        return text

    def are_abbreviations_similar(self, abbreviaion1, abbreviation2):
        if textdistance.levenshtein.normalized_similarity(abbreviaion1, abbreviation2) > 0.85:
            return True
        if abbreviaion1.lower() == abbreviation2.lower():
            return True
        return abbreviation2.startswith(abbreviaion1)

    def merge_similar_abbreviations(self, abbreviations_for_word):
        sorted_abbreviations = sorted(abbreviations_for_word.items(), key=lambda x:(-x[1],len(x[0]), x[0]))
        new_mapping = {}
        for i in range(len(sorted_abbreviations)):
            for j in range(i+1, len(sorted_abbreviations)):
                if self.are_abbreviations_similar(sorted_abbreviations[i][0], sorted_abbreviations[j][0]):
                    if sorted_abbreviations[j][0] not in new_mapping:
                        new_mapping[sorted_abbreviations[j][0]] = sorted_abbreviations[i][0] if sorted_abbreviations[i][0] not in new_mapping else new_mapping[sorted_abbreviations[i][0]]
            if sorted_abbreviations[i][0] not in new_mapping:
                new_mapping[sorted_abbreviations[i][0]] = sorted_abbreviations[i][0]
        new_abbreviations = {}
        for keyword in new_mapping:
            if new_mapping[keyword] not in new_abbreviations:
                new_abbreviations[new_mapping[keyword]] = 0
            new_abbreviations[new_mapping[keyword]] += abbreviations_for_word[keyword]
        return new_abbreviations

    def add_program_words_extensions(self):
        for abbr in self.resolved_abbreviations:
            new_mapping = {}
            for abbr_meaning in self.resolved_abbreviations[abbr]:
                for abbr_meaning_extended in set([re.sub(r"\bprogramme\b", "program", abbr_meaning), re.sub(r"\bprogram\b", "programme", abbr_meaning)]):
                    new_mapping[abbr_meaning_extended] = self.resolved_abbreviations[abbr][abbr_meaning]
            self.resolved_abbreviations[abbr] = new_mapping

    def resolve_abbreviations(self):
        self.find_abbreviations_in_hyponyms()
        self.resolved_abbreviations = {}
        for word in self.abbreviations:
            abbreviations_for_word = {}
            for abbr_pat in self.abbreviations[word]:
                text = abbr_pat[0] if abbr_pat[1] == word else abbr_pat[1]
                text_for_abbreviation = self.clean_abbreviation_expression(word, self.find_text_for_abbreviation(word, text))
                if text_for_abbreviation != "":
                    if text_for_abbreviation not in abbreviations_for_word:
                        abbreviations_for_word[text_for_abbreviation] = 0
                    abbreviations_for_word[text_for_abbreviation] += 1
            if len(abbreviations_for_word) > 0:
                self.resolved_abbreviations[word] = self.merge_similar_abbreviations(abbreviations_for_word)
        self.add_program_words_extensions()
        self.prepare_dictonaries_for_abbreviations()
        self.sorted_words_to_abbreviations = self.sort_word_expressions_for_abbreviations(self.words_to_abbreviations)
        self.sorted_resolved_abbreviations = self.sort_word_expressions_for_abbreviations(self.resolved_abbreviations)

    def sort_word_expressions_for_abbreviations(self, dict_to_compress):
        compressed_dictionary = {}
        for key in dict_to_compress:
            compressed_dictionary[key] = list(sorted(dict_to_compress[key].items(),key=lambda x:x[1], reverse=True))
        return compressed_dictionary

    def prepare_dictonaries_for_abbreviations(self):
        self.words_to_abbreviations = {}
        for abbr in self.resolved_abbreviations:
            for word_expr in self.resolved_abbreviations[abbr]:
                if word_expr not in self.words_to_abbreviations:
                    self.words_to_abbreviations[word_expr] = {}
                if abbr not in self.words_to_abbreviations[word_expr]:
                    self.words_to_abbreviations[word_expr][abbr] = 0
                self.words_to_abbreviations[word_expr][abbr] += self.resolved_abbreviations[abbr][word_expr]

    def save_dictionaries(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump([self.sorted_resolved_abbreviations, self.sorted_words_to_abbreviations], open(os.path.join(folder, "resolved_abbreviations.pickle"),"wb"))

    def find_abbreviation_resolving_by_meaning(self, text, abbr_meanings):
        if text in self.resolved_abbreviations:
            for abbr_meaning in abbr_meanings:
                if abbr_meaning in self.resolved_abbreviations[text]:
                    return abbr_meaning
        return self.sorted_resolved_abbreviations[text][0][0]

    def find_full_name(self, text, abbr_meanings):
        abbr, abbr_words = self.find_abbreviation_resolvings(text, abbr_meanings)
        if abbr == "":
            return text
        return "%s (%s)"%(abbr_words, abbr)

    def find_abbreviation_resolvings(self, text, abbr_meanings):
        abbr, abbr_words = "", ""
        if text in self.sorted_resolved_abbreviations:
            abbr_words = self.find_abbreviation_resolving_by_meaning(text, abbr_meanings)
            abbr = text
        elif text in self.sorted_words_to_abbreviations:
            abbr_words = text
            abbr = self.sorted_words_to_abbreviations[text][0][0]
        return abbr, abbr_words

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

    def replace_abbreviations(self, text, abbr_meanings=""):
        text_parts = deque(sorted(self.generate_subexpressions(text), key = lambda x: len(x.split()), reverse=True))
        new_text = text
        new_changed_text = text
        parsed_abbr_meanings = abbr_meanings.split(";")
        while len(text_parts) > 0:
            text_part = text_parts.popleft()
            res = self.find_full_name(text_part, parsed_abbr_meanings)
            if res != text_part:
                abbr, abbr_words = self.find_abbreviation_resolvings(text_part, parsed_abbr_meanings)
                if abbr_words in new_changed_text:
                    new_text = re.sub(r"\b%s\b"%abbr, " ", new_text)
                    new_changed_text = re.sub(r"\b%s\b"%abbr, " ", new_changed_text)
                new_text = re.sub(r"\b%s\b"%text_part, res, new_text)
                new_changed_text =  re.sub(r"\b%s\b"%text_part, "$", new_changed_text)
                text_parts = deque(sorted(self.generate_subexpressions(new_changed_text), key = lambda x: len(x.split()), reverse=True))
        return re.sub("\s+", " ", new_text)

    def find_abbreviations_meanings_in_text(self, text, abbr_meanings=""):
        text_parts = deque(sorted(self.generate_subexpressions(text), key = lambda x: len(x.split()), reverse=True))
        found_abbreviations = []
        new_changed_text = text
        parsed_abbr_meanings = abbr_meanings.split(";")
        while len(text_parts) > 0:
            text_part = text_parts.popleft()
            abbr, abbr_text = self.find_abbreviation_resolvings(text_part, parsed_abbr_meanings)
            if abbr != "":
                new_changed_text =  re.sub(r"\b%s\b"%text_part, "$", new_changed_text)
                found_abbreviations.append((abbr, abbr_text))
                text_parts = deque(sorted(self.generate_subexpressions(new_changed_text), key = lambda x: len(x.split()), reverse=True))
        return found_abbreviations
