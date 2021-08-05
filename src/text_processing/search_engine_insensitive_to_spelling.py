from utilities import utils
from text_processing import text_normalizer
import pickle
import re
import os
import pickle
from time import time
from text_processing import abbreviations_resolver

class SearchEngineInsensitiveToSpelling:
    
    def __init__(self, abbreviation_folder = "../model/abbreviations_dicts", load_abbreviations = False,
            symbols_count = 3, columns_to_process = ["title","abstract","keywords","identificators"]):
        self.dictionary_by_first_letters = {}
        self.id2docArray = []
        self.dictionary_small_words = {}
        self.newId = 0
        self.total_articles_number = 0
        self.symbols_count = symbols_count
        self._abbreviations_resolver = abbreviations_resolver.AbbreviationsResolver([])
        self.abbreviations_count_docs = {}
        self.docs_with_abbreviations = {}
        self.columns_to_process = columns_to_process
        self.load_abbreviations = load_abbreviations
        if self.load_abbreviations:
            self._abbreviations_resolver.load_model(abbreviation_folder)

    def calculate_abbreviations_count_docs(self, articles_df):
        self.docs_with_abbreviations = {}
        self.docs_with_abbreviaitons_by_id = {}
        for idx, abbr in enumerate(self._abbreviations_resolver.resolved_abbreviations):
            docs_found_for_abbr = self.find_articles_with_keywords([abbr], 1.0, extend_with_abbreviations=False)
            docs_resolved = set()
            for abbr_meaning in self._abbreviations_resolver.resolved_abbreviations[abbr]:
                docs_for_abbr_meaning = self.find_articles_with_keywords([abbr_meaning], 0.92, extend_with_abbreviations=False)
                for docId in docs_found_for_abbr.intersection(docs_for_abbr_meaning):
                    if docId not in self.docs_with_abbreviations:
                        self.docs_with_abbreviations[docId] = {}
                    if abbr not in self.docs_with_abbreviations[docId]:
                        self.docs_with_abbreviations[docId][abbr] = []
                    self.docs_with_abbreviations[docId][abbr].append(abbr_meaning)
                docs_resolved = docs_resolved.union(docs_for_abbr_meaning)
            for docId in docs_found_for_abbr - docs_resolved:
                if docId not in self.docs_with_abbreviations:
                    self.docs_with_abbreviations[docId] = {}
                if abbr not in self.docs_with_abbreviations[docId]:
                    self.docs_with_abbreviations[docId][abbr] = []
                self.docs_with_abbreviations[docId][abbr].append(self._abbreviations_resolver.sorted_resolved_abbreviations[abbr][0][0])
            if idx % 3000 == 0 or idx == len(self._abbreviations_resolver.resolved_abbreviations) - 1:
                print("Processed %d abbreviations"%idx)
        for docId in self.docs_with_abbreviations:
            for word in self.docs_with_abbreviations[docId]:
                if len(self.docs_with_abbreviations[docId][word]) > 1:
                    sorted_abbr = sorted(
                        [(w, self._abbreviations_resolver.resolved_abbreviations[word][w]) for w in self.docs_with_abbreviations[docId][word]],
                          key = lambda x: x[1], reverse =True)
                    self.docs_with_abbreviations[docId][word] = [sorted_abbr[0][0]]
        for docId in self.docs_with_abbreviations:
            for word in self.docs_with_abbreviations[docId]:
                abbr_meanings = set()
                for abbr_meaning in self.docs_with_abbreviations[docId][word]:
                    abbr_meanings.add(re.sub(r"\bprogramme\b", "program", abbr_meaning))
                    abbr_meanings.add(re.sub(r"\bprogram\b", "programme", abbr_meaning))
                self.docs_with_abbreviations[docId][word] = list(abbr_meanings)
        for docId in self.docs_with_abbreviations:
            art_id = articles_df["id"].values[docId] if "id" in articles_df.columns else docId
            for word in self.docs_with_abbreviations[docId]:
                if art_id not in self.docs_with_abbreviaitons_by_id:
                    self.docs_with_abbreviaitons_by_id[art_id] = {}
                self.docs_with_abbreviaitons_by_id[art_id][word] = self.docs_with_abbreviations[docId][word]
        self.abbreviations_count_docs = {}
        for i in self.docs_with_abbreviations:
            for key in self.docs_with_abbreviations[i]:
                if key not in self.abbreviations_count_docs:
                    self.abbreviations_count_docs[key] = {}
                for meaning in self.docs_with_abbreviations[i][key]:
                    if meaning not in self.abbreviations_count_docs[key]:
                        self.abbreviations_count_docs[key][meaning] = set()
                    self.abbreviations_count_docs[key][meaning].add(i)

    def save_model(self, folder="../model/search_index"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump([self.dictionary_by_first_letters, self.id2docArray, self.dictionary_small_words, self.total_articles_number, self.abbreviations_count_docs, self.docs_with_abbreviations, self.docs_with_abbreviaitons_by_id], open(os.path.join(folder, "search_index.pickle"),"wb"))

    def load_model(self, folder="../model/search_index"):
        self.dictionary_by_first_letters, self.id2docArray, self.dictionary_small_words, self.total_articles_number, self.abbreviations_count_docs, self.docs_with_abbreviations, self.docs_with_abbreviaitons_by_id = pickle.load(open(os.path.join(folder, "search_index.pickle"),"rb"))
    
    def create_inverted_index(self, articles_df, continue_adding = False, print_info = True):
        if continue_adding:
            self.total_articles_number += len(articles_df)
            self.unshrink_memory(set)
        else:
            self.total_articles_number = len(articles_df)
        for i in range(len(articles_df)):
            text = ""
            for column in self.columns_to_process:
                if column in ["keywords","identificators"]:
                    text = text + " . " + (text_normalizer.normalize_key_words_for_search(articles_df[column].values[i]) if column in articles_df.columns else "" )
                else:
                    text = text + " . " + text_normalizer.normalize_text(articles_df[column].values[i])
            text_words = text_normalizer.get_stemmed_words_inverted_index(text)
            for j in range(len(text_words)):
                self.add_item_to_dict(text_words[j], i)
                if j != len(text_words) - 1:
                    word_expression = text_words[j] + " " + text_words[j+1]
                    self.add_item_to_dict(word_expression, i)
            if print_info and (i % 20000 == 0 or i == len(articles_df) -1):
                print("Processed %d articles"%i)
        self.shrink_memory()
        if self.load_abbreviations:
            self.calculate_abbreviations_count_docs(articles_df)

    def shrink_memory(self, operation = list):
        for i in range(len(self.id2docArray)):
            self.id2docArray[i] = operation(self.id2docArray[i])
    
    def add_item_to_dict(self, word, docId):
        if len(word) == 0:
            return
        if len(word) < self.symbols_count:
            if word not in self.dictionary_small_words:
                self.dictionary_small_words[word] = self.newId 
                self.newId += 1
                self.id2docArray.append(set())
            self.id2docArray[self.dictionary_small_words[word]].add(docId)
            return
        if word[:self.symbols_count] not in self.dictionary_by_first_letters:
            self.dictionary_by_first_letters[word[:self.symbols_count]] = {}
        if word not in self.dictionary_by_first_letters[word[:self.symbols_count]]:
            self.dictionary_by_first_letters[word[:self.symbols_count]][word] = self.newId
            self.newId += 1
            self.id2docArray.append(set())
        self.id2docArray[self.dictionary_by_first_letters[word[:self.symbols_count]][word]].add(docId)

    def get_articles_by_word(self, word):
        try:
            if len(word) < self.symbols_count:
                return self.id2docArray[self.dictionary_small_words[word]]
        except:
            return []
        try:
            if len(word) >= self.symbols_count:
                return self.id2docArray[self.dictionary_by_first_letters[word[:self.symbols_count]][word]]
        except:
            return []
        return []

    def generate_sub_patterns(self, pattern):
        if pattern.strip() != "" and pattern.strip()[0] == "*":
            return ["*"]
        sub_patterns = set()
        res = ""
        cnt = 0
        for symb in pattern:
            if symb !="*":
                res += symb
                cnt += 1
            if symb == "*":
                res += "\w*"
                sub_patterns.add(res)
            if cnt == self.symbols_count:
                sub_patterns.add(res)
                break
        sub_patterns.add(res)
        return list(sub_patterns)

    def find_words_by_pattern(self, pattern):
        if re.search(r"\w+(\*\w*)+", pattern) is None:
            return [pattern]
        pattern = re.sub("[\*]+","*", pattern)
        new_pattern = pattern.replace("*","\w*")
        words_found = []
        if len(pattern.replace("*","")) < self.symbols_count:
            for w in self.dictionary_small_words:
                res = re.match(new_pattern, w)
                if res and res.group(0) == w and " " not in w:
                    words_found.append(w)
        if new_pattern[:self.symbols_count] in self.dictionary_by_first_letters:
            for key in self.dictionary_by_first_letters[new_pattern[:self.symbols_count]]:
                res = re.match(new_pattern, key)
                if res and res.group(0) == key and " " not in key:
                    words_found.append(key)
        else:
            for sub_pattern in self.generate_sub_patterns(pattern):
                for w in self.dictionary_by_first_letters:
                    res = re.match(sub_pattern, w)
                    if res and res.group(0) == w and " " not in w:
                        for key in self.dictionary_by_first_letters[w]:
                            res = re.match(new_pattern, key)
                            if res and res.group(0) == key and " " not in key:
                                words_found.append(key)
        return words_found
                
    def find_similar_words_by_spelling(self, word, threshold = 0.85, all_similar_words = False):
        time_total = time()
        stemmed_word = " ".join(text_normalizer.get_stemmed_words_inverted_index(word))
        stemmed_word = word if len(stemmed_word) < self.symbols_count else stemmed_word
        words = set([word, stemmed_word])
        if threshold >= 0.99 or len(stemmed_word) < self.symbols_count:
            return words
        words.add(re.sub(r"\bprogramme\b", "program", word))
        words.add(re.sub(r"\bprogram\b", "programme", stemmed_word))
        intial_words = words
        try:
            articles_count = len(self.get_articles_by_word(stemmed_word))
            for dict_word in self.dictionary_by_first_letters[word[:self.symbols_count]]:
                if all_similar_words or (articles_count == 0 or len(self.get_articles_by_word(dict_word)) < 4*articles_count):
                    for w in intial_words:
                        if utils.normalized_levenshtein_score(dict_word, w) >= threshold:
                            words.add(dict_word)
        except:
            pass
        z_s_replaced_words = set()
        for word in words:
            z_s_replaced_words = z_s_replaced_words.union(text_normalizer.replaced_with_z_s_symbols_words(word, self))
        return words.union(z_s_replaced_words)
    
    def find_keywords(self, stemmed_words):
        keywords = []
        if len(stemmed_words) > 2:
            for i in range(len(stemmed_words) - 1):
                keywords.append(stemmed_words[i] + " " + stemmed_words[i+1])
        else:
            keywords.append(" ".join(stemmed_words))
        return keywords
    
    def extend_query(self, query):
        words = query.split()
        prev_set = self.find_similar_words_by_spelling(words[0])
        for i in range(1, len(words)):
            new_set = set()
            for word in self.find_similar_words_by_spelling(words[i]):
                for prev_exp in prev_set:
                    new_set.add(prev_exp + " " + word)
            prev_set = new_set
        return prev_set

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

    def has_meaning_for_abbreviation(self, abbr_meanings, dict_to_check):
        return len(abbr_meanings.intersection(set([w[0] for w in dict_to_check]))) > 0

    def extend_with_abbreviations(self, query, dict_to_check, extend_abbr_meanings = "", add_to_meanings = False):
        abbr_meanings = set([w.strip() for w in extend_abbr_meanings.split(";") if w.strip() != ""])
        new_queries = set([query])
        subexpressions = self.generate_subexpressions(query)
        for expr in subexpressions:
            if expr in dict_to_check:
                if self.has_meaning_for_abbreviation(abbr_meanings, dict_to_check[expr]):
                    for word,cnt in dict_to_check[expr]:
                        if expr in word:
                            continue
                        if word in abbr_meanings:
                            new_query = re.sub(r"\b%s\b"%expr,  word, query)
                            if new_query not in new_queries:
                                if add_to_meanings and expr.strip() != "":
                                    abbr_meanings.add(expr)
                                extend_abbr_meanings = ";".join(list(abbr_meanings))
                                new_queries_part, extend_abbr_meanings  = self.extend_with_abbreviations(new_query, dict_to_check, extend_abbr_meanings)
                                new_queries = new_queries.union(new_queries_part)
                else:
                    for word,cnt in dict_to_check[expr]:
                        if expr in word or (len(dict_to_check[expr]) > 1 and cnt < 15):
                            continue
                        new_query = re.sub(r"\b%s\b"%expr,  word, query)
                        if new_query not in new_queries:
                            if add_to_meanings and expr.strip() != "":
                                abbr_meanings.add(expr)
                            extend_abbr_meanings = ";".join(list(abbr_meanings))
                            new_queries_part, extend_abbr_meanings  = self.extend_with_abbreviations(new_query, dict_to_check, extend_abbr_meanings)
                            new_queries = new_queries.union(new_queries_part)
        return new_queries, extend_abbr_meanings

    def extend_query_with_abbreviations(self, query, extend_with_abbreviations, extend_abbr_meanings=""):
        if not extend_with_abbreviations:
            return set(), extend_abbr_meanings
        normalized_key = text_normalizer.normalize_text(query)
        extended_queries = set()
        extended_queries_part, extend_abbr_meanings = self.extend_with_abbreviations(normalized_key, self._abbreviations_resolver.sorted_resolved_abbreviations, extend_abbr_meanings)
        extended_queries = extended_queries.union(extended_queries_part)
        new_extended_queries = set(extended_queries)
        for new_query in extended_queries:
            new_extended_queries_part, extend_abbr_meanings = self.extend_with_abbreviations(new_query, self._abbreviations_resolver.sorted_words_to_abbreviations, extend_abbr_meanings, add_to_meanings = True)
            new_extended_queries = new_extended_queries.union(new_extended_queries_part)
        return new_extended_queries, extend_abbr_meanings

    def get_article_with_special_abbr_meanings(self, query, abbr_meanings):
        if abbr_meanings.strip() == "" or len([w for w in query.split() if text_normalizer.is_abbreviation(w)]) == 0:
            return set(), False
        docs_with_abbreviations = set()
        first_assignment = True
        for abbr_meaning in abbr_meanings.split(";"):
            abbr_meaning = abbr_meaning.strip()
            if abbr_meaning not in self._abbreviations_resolver.sorted_words_to_abbreviations:
                continue
            for word,cnt in self._abbreviations_resolver.sorted_words_to_abbreviations[abbr_meaning]:
                if re.search(r"\b%s\b"%word, query) != None and word in self.abbreviations_count_docs and abbr_meaning in self.abbreviations_count_docs[word]:
                    if first_assignment:
                        docs_with_abbreviations = docs_with_abbreviations.union(self.abbreviations_count_docs[word][abbr_meaning]) 
                        first_assignment = False
                    else:
                        docs_with_abbreviations = docs_with_abbreviations.intersection(self.abbreviations_count_docs[word][abbr_meaning])
        return docs_with_abbreviations, True
    
    def find_articles_with_keywords(self, key_words, threshold = 0.85, extend_query = False, extend_with_abbreviations = True, extend_abbr_meanings = ""):
        total_articles = set()
        time_start = time()
        time_total = time()
        for key in key_words:
            normalized_key = text_normalizer.normalize_text(key)
            extended_queries = self.extend_query(normalized_key) if extend_query else set([normalized_key])
            extended_queries_with_abbr, extend_abbr_meanings = self.extend_query_with_abbreviations(key,extend_with_abbreviations, extend_abbr_meanings)
            extended_queries = extended_queries.union(extended_queries_with_abbr)
            time_start = time()
            for query in extended_queries:
                first_assignment = True
                articles = set()
                for key_word in self.find_keywords(text_normalizer.get_stemmed_words_inverted_index(query)):
                    sim_word_articles = set()
                    for sim_word in self.find_similar_words_by_spelling(key_word, threshold):
                        sim_word_articles = sim_word_articles.union(set(self.get_articles_by_word(sim_word)))
                    if first_assignment:
                        articles = articles.union(sim_word_articles)
                        first_assignment = False
                    else:
                        articles = articles.intersection(sim_word_articles)
                docs_with_abbreviations, has_abbr = self.get_article_with_special_abbr_meanings(query, extend_abbr_meanings)
                if not has_abbr:
                    total_articles = total_articles.union(articles)
                else:
                    total_articles = total_articles.union(articles.intersection(docs_with_abbreviations))
                time_start = time()
        return total_articles

    def find_articles_with_keywords_extended(self, key_words, threshold = 0.9, extend_query = False, extend_with_abbreviations = True, extend_abbr_meanings = ""):
        full_keywords = set()
        for query in key_words:
            words = query.split()
            prev_set = set(self.find_words_by_pattern(words[0]))
            for i in range(1, len(words)):
                new_set = set()
                for word in self.find_words_by_pattern(words[i]):
                    for prev_exp in prev_set:
                        new_set.add(prev_exp + " " + word)
                prev_set = new_set
            full_keywords = full_keywords.union(prev_set)
        return self.find_articles_with_keywords(list(full_keywords), threshold = threshold, extend_query = extend_query,\
         extend_with_abbreviations = extend_with_abbreviations, extend_abbr_meanings = extend_abbr_meanings)

    def save_diminished_dictionary_for_synonyms_app(self, folder):
        dictionary_by_first_letters = {}
        for key in self.dictionary_by_first_letters:
            if len(key) <= 2:
                if len(self.id2docArray[self.dictionary_small_words[key]]) >= 2:
                    dictionary_by_first_letters[key] = len(self.id2docArray[self.dictionary_small_words[key]])
            else:
                dictionary_by_first_letters[key] = {}
                for key_word in self.dictionary_by_first_letters[key]:
                    if len(self.id2docArray[self.dictionary_by_first_letters[key][key_word]]) >= 2:
                        dictionary_by_first_letters[key][key_word] = len(self.id2docArray[self.dictionary_by_first_letters[key][key_word]])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump([dictionary_by_first_letters, self.total_articles_number, {}], open(os.path.join(folder, "search_index.pickle"),"wb"))