from text_processing import text_normalizer
from text_processing import search_engine_insensitive_to_spelling
import re
import nltk

class AdvancedTextNormalizer:

    def __init__(self, abbreviations_resolver, full_normalization = False, status_logger = None):
        self.full_normalization = full_normalization
        self.abbreviations_resolver = abbreviations_resolver
        self.status_logger = status_logger

    def clean_uppercase(self, articles_df, columns_to_clean):
        temp_search_engine_inverted_index = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling(columns_to_process = ["abstract"])
        temp_search_engine_inverted_index.create_inverted_index(articles_df)
        all_search_engine_inverted_index = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling()
        all_search_engine_inverted_index.create_inverted_index(articles_df)
        articles_df = self.replace_uppercase_words(articles_df, columns_to_clean, temp_search_engine_inverted_index, all_search_engine_inverted_index, all_search_engine_inverted_index)
        return articles_df

    def clean_glued_words(self, articles_df, columns_to_clean):
        search_engine_inverted_index = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling()
        search_engine_inverted_index.create_inverted_index(articles_df)
        articles_df = self.separate_glued_pairs(articles_df, columns_to_clean, search_engine_inverted_index, search_engine_inverted_index)
        return articles_df

    def are_words_glued(self, word, stop_word, search_engine_inverted_index):
        if " " in word or len(stop_word) == 1 or (len(word)-len(stop_word)) < 4 or len(search_engine_inverted_index.get_articles_by_word(word)) > 5:
            return False
        lemmatized_word = text_normalizer.lemmatize_word(word[len(stop_word):])
        if lemmatized_word == "ever":
            return False
        if word[:len(stop_word)] == stop_word and len(search_engine_inverted_index.get_articles_by_word(lemmatized_word)) > 70:
            return True
        lemmatized_word = text_normalizer.lemmatize_word(word[:-len(stop_word)])
        if word[-len(stop_word):] == stop_word and len(search_engine_inverted_index.get_articles_by_word(lemmatized_word)) > 70:
            return True
        return False

    def find_glued_words(self, search_engine_inverted_index, big_search_engine_inverted_index):
        stop_word_for_glued_words = set(nltk.corpus.stopwords.words("english")) - set(["me","he","she","her","it","am","be","do","an",\
            "or","as","at","by","up","in","out","un","on","off","over","under","all","any","no","nor","so","don","re","ll","ve","ain","ma","shan","down","for"])
        exception_words = ["beforehand","profits", "hereinafter","wholesome","topics", "discours", "meanwhile", "goodwill","forages","foresters", "forage","forester","wellbeing"]

        pairs_with_glued_words = {}
        for key in search_engine_inverted_index.dictionary_by_first_letters:
            if len(key) > 2:
                for word in search_engine_inverted_index.dictionary_by_first_letters[key]:
                    for stop_word in stop_word_for_glued_words:
                        if word not in exception_words and self.are_words_glued(word, stop_word,big_search_engine_inverted_index):
                            print(stop_word, word)
                            for doc in search_engine_inverted_index.get_articles_by_word(word):
                                if doc not in pairs_with_glued_words:
                                    pairs_with_glued_words[doc] = []
                                pairs_with_glued_words[doc].append((word, stop_word))
                            break
        return pairs_with_glued_words

    def clean_text(self, text, text_to_replace, stop_word):
        while re.search(text_to_replace, text, flags=re.IGNORECASE) != None:
            old_val = re.search(text_to_replace, text, flags=re.IGNORECASE).group(0)
            if old_val.isupper():
                break
            new_val = ""
            if re.match(stop_word, old_val[:len(stop_word)], flags=re.IGNORECASE):
                new_val = old_val[:len(stop_word)] + " " + old_val[len(stop_word):]
            elif re.match(stop_word, old_val[-len(stop_word):], flags=re.IGNORECASE):
                new_val = old_val[:-len(stop_word)] + " " + old_val[-len(stop_word):]
            if old_val[0].isupper() and old_val[1:].islower():
                break
            text = text.replace(old_val, new_val)
            print(old_val, "###",new_val,"###",text_to_replace, "###",stop_word)
        return text

    def separate_glued_pairs(self, articles_df, columns_to_clean, search_engine_inverted_index, big_search_engine_inverted_index):
        pairs_with_glued_words = self.find_glued_words(search_engine_inverted_index, big_search_engine_inverted_index)
        for doc in pairs_with_glued_words:
            for pair in pairs_with_glued_words[doc]:
                for column in columns_to_clean:
                    articles_df[column].values[doc] = self.clean_text(articles_df[column].values[doc], pair[0],pair[1])
        return articles_df

    def replace_uppercase_words(self, articles_df, columns_to_clean, temp_search_engine_inverted_index, all_search_engine_inverted_index, search_engine_inverted_index):
        words_set = set()
        for key in all_search_engine_inverted_index.dictionary_by_first_letters:
            if len(key) >= 2:
                words_to_process = all_search_engine_inverted_index.dictionary_by_first_letters[key] if type(all_search_engine_inverted_index.dictionary_by_first_letters[key]) == dict else [key]
                for word in words_to_process:
                    if word.isupper() and not word in self.abbreviations_resolver.resolved_abbreviations:
                        for w in word.split():
                            if w not in self.abbreviations_resolver.resolved_abbreviations and len(w) > 1 and re.search("\d+",w) == None:
                                words_set.add(w)
        words_to_change = []
        for word in words_set:
            w_lem = text_normalizer.lemmatize_word(word.lower())
            w_count = len(temp_search_engine_inverted_index.get_articles_by_word(w_lem))
            w_up_count = len(temp_search_engine_inverted_index.get_articles_by_word(word))
            if (w_up_count == 0 and w_count > 0) or (w_up_count > 0 and w_count/w_up_count > 3):
                words_to_change.append(word)
        for word in words_to_change:
            for article_id in search_engine_inverted_index.get_articles_by_word(word):
                for column in columns_to_clean:
                    articles_df[column].values[article_id] = re.sub(r"\b%s\b"%word, word.lower(), articles_df[column].values[article_id])
        for word in text_normalizer.stopwords_all:
            if word.upper() in self.abbreviations_resolver.resolved_abbreviations:
                continue
            for article_id in search_engine_inverted_index.get_articles_by_word(word.upper()):
                for column in columns_to_clean:
                    articles_df[column].values[article_id] = re.sub(r"\b%s\b"%word.upper(), word.lower(), articles_df[column].values[article_id])
        return articles_df

    def normalize_key_word_columns(self, articles_df, key_words_columns, normalize = True):
        for i in range(len(articles_df)):
            for column in key_words_columns:
                if column in articles_df.columns:
                    text = articles_df[column].values[i]
                    text = ";".join([key_word for key_word in text.replace(",",";").split(";") if "full text" not in key_word.lower() and key_word.strip() != ""])
                    key_words = [w.strip() for w in text.split(";") if w.strip() != ""]
                    if len(key_words) > 5 and (len([w for w in key_words if w.strip().isupper()]) / len(key_words)) >= 0.5:
                        text = ";".join([(w.lower() if w.isupper() else w) for w in key_words])
                    articles_df[column].values[i] = text
                    if normalize:
                        text = ";".join([text_normalizer.normalize_text(key_word, False) for key_word in text.replace(",",";").split(";")])
                    articles_df[column].values[i] = text.strip()
        return articles_df

    def normalize_text_columns(self, articles_df, key_words_columns):
        for column in ["title", "abstract"]:
            articles_df[column] = articles_df[column].apply(text_normalizer.remove_unicode_symbols)
            articles_df[column] = articles_df[column].apply(text_normalizer.clean_semicolon_expressions)
            articles_df[column] = articles_df[column].apply(text_normalizer.clean_semicolon_expressions_in_the_end)
            articles_df[column] = articles_df[column].apply(text_normalizer.replace_brackets_with_signs)
            articles_df[column] = articles_df[column].apply(text_normalizer.remove_all_last_square_brackets_from_text)
            articles_df[column] = articles_df[column].apply(text_normalizer.remove_html_tags)
            articles_df[column] = articles_df[column].apply(text_normalizer.clean_text_from_commas)
            articles_df[column] = articles_df[column].apply(text_normalizer.remove_copyright)
            articles_df[column] = articles_df[column].apply(text_normalizer.clean_text_from_commas)

        articles_df = self.normalize_key_word_columns(articles_df, key_words_columns, normalize = True)
        return articles_df

    def log_percents(self, percent):
        if self.status_logger is not None:
            self.status_logger.update_step_percent(percent)

    def normalize_text_for_df(self, articles_df, key_words_columns = ["keywords", "identificators"], columns_to_clean = ["title", "abstract", "keywords", "identificators"]):
        articles_df = self.normalize_text_columns(articles_df, key_words_columns)
        self.log_percents(25)
        if self.full_normalization:
            articles_df = self.clean_uppercase(articles_df, columns_to_clean)
            self.log_percents(50)
            articles_df = self.clean_glued_words(articles_df, columns_to_clean)
            self.log_percents(75)
            articles_df = self.normalize_text_columns(articles_df, key_words_columns)
        return articles_df

