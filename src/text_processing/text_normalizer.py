from langdetect import detect
import unicodedata
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pycountry
import pandas as pd
import textdistance
import os

stop_word_units = ["kg", "g", "mg", "ml", "l", "s", "ms", "km", "mm","la","en","le","de","et","los","une", "un", "del",\
"i","ii","iii","iv","A","per","also","ha","cm","non","ton","etc","el", "among", "along"]
stopwords_all = set(nltk.corpus.stopwords.words("english") + stop_word_units)
lemma_exception_words = {"assess":"assess", "assesses":"assess"}
lmtzr = WordNetLemmatizer()


def get_all_freq_words(filename = '../data/MostFrequentWords.xlsx'):
    if not os.path.exists(filename):
        return {}, {}
    freq_words = {}
    ranks = {}
    temp_df = pd.read_excel(filename).fillna("")
    for i in range(len(temp_df)):
        word = temp_df["Word"].values[i].lower().strip() 
        if word not in freq_words:
            freq_words[word] = []
            ranks[word] = 100000
        freq_words[word].append(temp_df["Part of speech"].values[i].strip())
        ranks[word] = min(ranks[word], temp_df["Rank"].values[i])
    return freq_words, ranks

freq_words, ranks = get_all_freq_words()

def get_rank_of_word(word):
    rank = 5000
    if word in ranks:
        rank = min(rank, ranks[word])
    if re.sub(r"ing\b", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"ing\b", "", word)])
    if re.sub(r"ing\b", "e", word) in ranks:
        rank = min(rank, ranks[re.sub(r"ing\b", "e", word)])
    if re.sub(r"s\b", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"s\b", "", word)])
    if re.sub(r"es\b", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"es\b", "", word)])
    if re.sub(r"ed\b", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"ed\b", "", word)])
    if re.sub(r"d\b", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"d\b", "", word)])
    if len(word) < 7:
        return rank / 5000
    if re.sub(r"\bin", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"\bin", "", word)])
    if re.sub(r"\bun", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"\bun", "", word)])
    if re.sub(r"\bim", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"\bim", "", word)])
    if re.sub(r"\bir", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"\bir", "", word)])
    if re.sub(r"\bsub", "", word) in ranks:
        rank = min(rank, ranks[re.sub(r"\bsub", "", word)])
    return rank / 5000

def remove_unicode_symbols(text):
    res = re.search("<\s*U\s*\+([\w\d]{4})\s*>", text)
    if res:
        text = re.sub("<\s*U\s*\+([\w\d]{4})\s*>", " ; ", text)
        text = re.sub("([\.:;!\?])\s*;", "\g<1>", text)
        text = re.sub("^\s*;\s*", "", text)
        text = re.sub("\s*;\s*", " ; ", text)
    return text

def calculate_common_topic_score(concept):
    score = 0
    for word in concept.split():
        score += get_rank_of_word(word)
    return score/ len(concept.split()) if len(concept.split()) > 0 else 0

def get_only_english_text(raw_text, separator):
    text = []
    for text_part in raw_text.split(separator):
        try:
            if text_part.strip()  != "" and detect(text_part.lower()) == "en":
                text.append(text_part)
        except:
            print(text_part)
            text.append(text_part)
    new_text = separator.join(text).strip()
    return  new_text if new_text != "" else raw_text

def lemmatize_word(word):
    return lmtzr.lemmatize(word)

def get_bigrams(phrase, phrases_model):
    split_phrase = phrase.split()
    res = set()
    for i in range(len(split_phrase)):
        if i > 0 and "_" in phrases_model[[split_phrase[i-1],split_phrase[i]]][0]:
            res.add(split_phrase[i-1] + " " + split_phrase[i])
        res.add(split_phrase[i])
    return res

def update_title_and_abstract_only_english(df, abstr_separator = ";", title_separator = ";"):
    for i in range(len(df)):
        lower_abstract = df["abstract"].values[i].lower()
        lower_title = df["title"].values[i].lower()
        if abstr_separator in df["abstract"].values[i] and detect(lower_abstract) != "en":
            print(i)
            print(df["abstract"].values[i])
            df["abstract"].values[i] = get_only_english_text(df["abstract"].values[i], abstr_separator)
            print(df["abstract"].values[i])
            print("######")
        if title_separator in df["title"].values[i] and detect(lower_title) != "en":
            print(i)
            print(df["title"].values[i])
            df["title"].values[i] = get_only_english_text(df["title"].values[i], title_separator)
            print(df["title"].values[i])
            print("###")
    return df

def split_sentence_to_parts(sentence, remove_and_or=True):
    if remove_and_or:
        sentence = re.sub(r"\b(and|or)\b", ",", sentence)
    sentence = re.sub(r"\(\s*[\w\d]{1,4}\s*\)", ",", sentence)
    sentence = re.sub(r"\[\s*[\w\d]{1,4}\s*\]", ",", sentence)
    sentence = re.sub(r"\(|\)|\[|\]|;|:", ",", sentence)
    return [part.strip() for part in sentence.split(",") if part.strip()]

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def get_normalized_text_with_numbers(document):
    stemmer = SnowballStemmer("english")
    lmtzr = WordNetLemmatizer()
    document = document.lower()
    words = []
    reg = re.compile("[^a-zA-Z0-9]")
    document = reg.sub(" ", document).strip()
    for word in document.split():
        if word not in stopwords_all and (len(word) > 1 or word in "0123456789"):
            lemmatized_word = lmtzr.lemmatize(word)
            stemmed_word = stemmer.stem(lemmatized_word)
            words.append(stemmed_word)
    return words

def is_abbreviation(text):
    if text.isupper():
        return True
    if len(text) > 1 and text[-1] == "s" and text[:-1].isupper():
        return True
    return False

def normalize_abbreviation(text):
    if text[-1] == "s":
        return text[:-1]
    return text

def normalize_text(text, lower=True):
    reg = re.compile(r"[^\w\d]|\b\d+\w*\b|_")
    text = reg.sub(" ", text).strip()
    text = re.sub("\s+", " ", text)
    text = remove_accented_chars(text)
    text = remove_unicode_symbols(text)
    if lower:
        new_text = ""
        for word in text.split():
            if is_abbreviation(word):
                new_text = new_text + " " + normalize_abbreviation(word)
            else:
                new_text = new_text + " " + word.lower()
        text = new_text
    return text.strip()

def remove_copyright(text):
    text = text if text.replace("[","").replace("]","").lower() != "no abstract available" else ""
    return re.sub("\s+"," ", re.sub("([C|c])opyright ?\.?", "", text.split("©")[0].strip().split("(c)")[0].strip().split("(C)")[0].strip()).strip(), flags=re.IGNORECASE)

def get_stemmed_words_inverted_index(doc, regex = r'\b\w+\b', to_lower = True, lemmatize_capital_words = True):
    words = []
    for w in doc.split(" "):
        list_w = re.findall(regex, w)
        for list_word in list_w: 
            if list_word in stopwords_all or re.match(r"\d+", list_word) != None:
                continue
            if is_abbreviation(list_word):
                lemmatized_word = normalize_abbreviation(list_word)
                if lemmatized_word not in stopwords_all:
                    words.append(lemmatized_word)
            else:
                if len(list_word) > 1 and list_word.lower() not in stopwords_all:
                    lemmatized_word = list_word
                    if lemmatize_capital_words:
                        lemmatized_word = lmtzr.lemmatize(list_word)
                    else:
                        lemmatized_word = lmtzr.lemmatize(list_word) if list_word[0].islower() else list_word
                    if list_word in lemma_exception_words:
                        lemmatized_word = lemma_exception_words[list_word]
                    if to_lower:
                        lemmatized_word = lemmatized_word.lower()
                    if(len(lemmatized_word) <= 1):
                        lemmatized_word = list_word
                    if lemmatized_word not in stopwords_all:
                        words.append(lemmatized_word)
    return words

def get_all_freq_words():
    freq_words = set()
    temp_df = pd.read_excel('../data/MostFrequentWords.xlsx').fillna("")
    for i in range(len(temp_df)):
        word = temp_df["Word"].values[i].lower().strip() 
        freq_words.add(word)
    return freq_words

def build_filter_dictionary(filenames = ["../data/Filter_Geo_Names.xlsx"]):
    filter_word_list = []
    for filename in filenames:
        temp_df = pd.read_excel(filename).fillna("")
        for column_name in temp_df.columns:
            for i in range(len(temp_df)):
                normalized_phr = normalize_sentence(temp_df[column_name].values[i]).strip()
                if len(temp_df[column_name].values[i].strip()) == 0 or len(normalized_phr) == 0:
                    continue
                filter_word_list.append(temp_df[column_name].values[i].strip())
                filter_word_list.append(normalized_phr)
    return list(set(filter_word_list))

def normalize_sentences(texts):
    sentences =[get_stemmed_words_inverted_index(normalize_text(text)) for text in texts]
    return sentences

def normalize_sentence(text):
    return " ".join(get_stemmed_words_inverted_index(normalize_text(text)))

def normalize_sentence_after_phrases(sentence):
    return [word.replace("_"," ") for word in sentence]

def normalize_country_name(country_name):
    country_name = country_name.split(',')[0].strip()
    country_name = re.sub("\s+", " ", re.sub("\(.*?\)|\[.*?\]", " ", country_name)).strip()
    return remove_accented_chars(country_name)

def tokenize_words(text):
    return list(filter(None, re.split(r"[,;.!?:\(\)]+|\band\b|\bor\b", text)))

def get_phrased_sentence(text, phrases, phrases_3gram=None):
    sentence = []
    for sent in normalize_sentences(tokenize_words(text)):
        if phrases is None:
            sentence.extend(sent)
        else:
            sentence.extend(normalize_sentence_after_phrases(
                phrases[sent] if phrases_3gram == None else phrases_3gram[phrases[sent]]))
    return sentence

def clean_semicolon_expressions(text):
    new_text = ""
    add_short_semicolon_expressions = False
    was_changed =False
    if text == "" or ";" not in text:
        return text
    splitted_expressions = text.split(";")
    for idx, expr in enumerate(splitted_expressions):
        if re.match(r"[\w\d -]*", expr) != None and re.match(r"[\w\d -]*", expr).group(0) == expr and len(expr.split()) < 2 and not add_short_semicolon_expressions:
            if idx < len(splitted_expressions) - 1 and splitted_expressions[idx+1].strip() != "" and splitted_expressions[idx+1].strip()[0].islower():
                add_short_semicolon_expressions = True
            was_changed = True
        else:
            add_short_semicolon_expressions = True
        if add_short_semicolon_expressions:
            if new_text == "":
                new_text += expr
            else:
                new_text = new_text + ";" +expr
    if was_changed:
        return new_text.strip()
    return text

def clean_semicolon_expressions_in_the_end(text):
    new_text = ""
    add_short_semicolon_expressions = False
    was_changed =False
    if text == "":
        return text
    splitted_expressions = text.split(";")
    for idx, expr in enumerate(splitted_expressions):
        if len(expr.split()) > 3:
            if new_text == "":
                new_text += expr
            else:
                new_text = new_text + ";" + expr
        if expr.strip() != "" and expr.strip()[-1] == "." and idx < len(splitted_expressions) -1 and len(splitted_expressions[idx+1].split()) < 3:
            was_changed =True
            break
    if was_changed:
        return new_text
    return text

def contain_name(word, words_to_check):
    for w in words_to_check:
        if w in word:
            return True
    return False

def contain_full_name(word, words_to_check):
    for w in words_to_check:
        if re.search(r"\b%ss?\b"%w,word) != None:
            return True
    return False

def contain_verb_form(verb, verbs_to_check):
    for v in verbs_to_check:
        if v in verb or re.sub(r"e\b","",v) in verb:
            return True
    return False

def has_word_with_one_non_digit_symbol(word):
    for w in word.split():
        if re.search("\d",w) != None:
            word_norm = re.sub("\d","", w)
            if len(word_norm) < 2:
                return True
    return False

def has_word_in_sentence(sentence, words):
    sent_split = set([w.lower() for w in sentence.split()])
    for word in words:
        if word in sent_split:
            return True
    return False

def are_words_similar_by_symbol_replacement(first_word, symbol, symbol_to_replace, second_word):
    if symbol in first_word and len(first_word) > 5:
        text = first_word
        while text != "":
            try:
                ind = text.rindex(symbol)
                if ind < len(first_word) - 1 and ind > 1:
                    new_word = first_word[:ind] + symbol_to_replace + first_word[ind+1:]
                    if new_word == second_word:
                        return True
                text = text[:ind]
            except:
                text = ""
    return False

def are_words_similar_by_replacing_z_s(first_word, second_word):
    return are_words_similar_by_symbol_replacement(first_word, "s","z",second_word) or are_words_similar_by_symbol_replacement(first_word, "z","s",second_word)

def are_words_similar(first_word, second_word, threshold):
    if textdistance.levenshtein.normalized_similarity(first_word, second_word) >= threshold:
        return True
    return are_words_similar_by_replacing_z_s(first_word, second_word)

def normalize_without_lowering(word_expr):
    words = []
    for word in word_expr.split():
        if word.lower() not in stopwords_all and lmtzr.lemmatize(word.lower()) not in stopwords_all:
            words.append(word)
    return " ".join(words)

def replace_string_default_values(articles_df, column_name):
    for article in range(len(articles_df)):
        if articles_df[column_name].values[article] == "":
            articles_df[column_name].values[article] = []
        else:
            articles_df[column_name].values[article] = list(articles_df[column_name].values[article])
    return articles_df

def replace_brackets_with_signs(text):
    pat_set = set()
    for m in re.finditer("\{.*?\}", text):
        match_str = m.group(0)
        pat_set.add(match_str)
    dict_patterns_to_change = {}
    for pat in pat_set:
        if re.search("\w+|[%&#]",pat) != None:
            res = re.search("\w+|[%&#]",pat).group(0)
            if len(res) == 1 and res != "_":
                dict_patterns_to_change[pat] = res
        if pat not in dict_patterns_to_change:
            dict_patterns_to_change[pat] = ""
    for key in dict_patterns_to_change:
        text = text.replace(key, dict_patterns_to_change[key])
    return text

def remove_html_tags(text):
    text = re.sub(r"<.+?>","",text)
    text = re.sub(r"</.+?>","",text)
    text = re.sub(r"\(\s*\)","",text)
    text = re.sub(r"\[\s*\]","",text)
    text = text.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<").replace("&quot;","\"").replace("&nbsp;", " ")
    text = re.sub(r"&\w+;", " ", text)
    return re.sub("\s+"," ",text)

def remove_last_square_brackets(text):
    sq_brackets = re.findall(r"\[.*?\]",text)
    if len(sq_brackets) ==0:
        return text
    if len(get_stemmed_words_inverted_index(text.replace(sq_brackets[-1],"").strip())) < 2:
        return text.replace("["," ").replace("]"," ").strip()
    if len(get_stemmed_words_inverted_index(text.split(sq_brackets[-1])[1])) > 2:
        return text
    return text.replace(sq_brackets[-1],"").strip()

def remove_all_last_square_brackets_from_text(text):
    new_text = remove_last_square_brackets(text)
    while  new_text != text:
        text = new_text
        new_text = remove_last_square_brackets(text)
    return new_text

def get_similar_word_by_symbol_replacement(word, symbol, symbol_to_replace, search_engine_inverted_index):
    if symbol in word and len(word) > 5 and " " not in word:
        text = word
        while text != "":
            try:
                ind = text.rindex(symbol)
                if ind < len(word) - 1 and ind > 1:
                    new_word = word[:ind] + symbol_to_replace + word[ind+1:]
                    if len(search_engine_inverted_index.get_articles_by_word(new_word)) > 0:
                        return new_word
                text = text[:ind]
            except:
                text = ""
    return ""

def replaced_with_z_s_symbols_words(word, search_engine_inverted_index):
    new_words = set()
    new_word = get_similar_word_by_symbol_replacement(word, "s","z",search_engine_inverted_index)
    if new_word.strip() != "":
        new_words.add(new_word)
    new_word = get_similar_word_by_symbol_replacement(word, "z","s",search_engine_inverted_index)
    if new_word.strip() != "":
        new_words.add(new_word)
    return new_words

def clean_text_from_commas(word, marks = ",.;?!"):
    ind = 0
    while ind < len(word):
        if word[ind] in marks or word[ind:ind+1].strip() == "":
            ind +=1
        else:
            break
    word = word[ind:]
    ind = len(word) -1
    last_sym = ""
    while ind >= 0:
        if word[ind] in marks or word[ind:ind+1].strip() == "":
            ind -=1
            if word[ind] in ".?!":
                last_sym = word[ind]
        else:
            break
    word = word[:ind+1] + last_sym
    return word

def normalize_key_words_for_search(text, normalize = True, to_lower = False, _remove_accented_chars = False):
    text = ";".join([key_word for key_word in text.replace(",",";").split(";") if "full text" not in key_word.lower() and key_word.strip() != ""])
    if normalize:
        text = ";".join([normalize_text(key_word, to_lower) for key_word in text.replace(",",";").split(";")])
    if _remove_accented_chars:
        text = remove_accented_chars(text)
    return " ; ".join([word.strip() for word in text.split(";") if word.strip() != ""])