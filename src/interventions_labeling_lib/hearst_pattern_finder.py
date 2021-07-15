import re
import string
import nltk
from nltk.tag.perceptron import PerceptronTagger
import spacy
from text_processing import text_normalizer
from nltk.stem.wordnet import WordNetLemmatizer
from geotext import GeoText
import langdetect

nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()


class HearstPatterns(object):

    def __init__(self, extended = False, for_finding_abbreviations = False, for_finding_comparison = False):
        self.__chunk_patterns = r""" #  helps us find noun phrase chunks
                NP: {<DT|PRP\$>?<JJ>*<NN>+}
                    {<DT|PRP\$>?<NNP>+}
                    {<DT|PRP\$>?<NNS>+}
        """

        self.__np_chunker = nltk.RegexpParser(self.__chunk_patterns)
        
        self.__hearst_patterns = [
                ("(NP_\w+ (, )?such as (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?including (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)", "first")
            ]

        if extended:
            self.__hearst_patterns.extend([
                ("NP_\w+ ?@ ?(NP_\w+ ?(, )?(and |or )?)+ ?@", "first"),
                ("(NP_\w+ ?(, )?# ?(NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?any other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?some other NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?is NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?was NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?were NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?are NP_\w+)", "last"),
                ("(NP_\w+ (, )?like (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?like other NP_\w+)", "last"),
                ("examples of (NP_\w+ (, )?is (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("examples of (NP_\w+ (, )?are (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?are examples of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?is example of NP_\w+)", "last"),
                ("(NP_\w+ (, )?for example (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?which is called NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?which is named NP_\w+)", "last"),
                ("(NP_\w+ (, )?mainly (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?mostly (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?notably (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?particularly (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?principally (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?in particular (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?except (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?other than (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?e.g. (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?i.e. (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?kind of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?kinds of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?form of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?forms of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?which looks like NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?which sounds like NP_\w+)", "last"),
                ("(NP_\w+ (, )?which are similar to (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?which is similar to (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?examples of this is (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?examples of this are (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?type of NP_\w+)", "last"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?types of NP_\w+)", "last"),
                ("(NP_\w+ (, )?types (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )? NP_\w+ types)", "last"),
                ("(compare (NP_\w+ ?(, )?)+(and |or )?with NP_\w+)", "last"),
                ("(NP_\w+ (, )?compared to (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("(NP_\w+ (, )?among them (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("((NP_\w+ ?(, )?)+(and |or )?(is |are |were |was )?sort of NP_\w+)", "last"),
            ])
        if for_finding_comparison:
            self.__hearst_patterns = [ ("([Cc]omparison of (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("([Cc]omparison between (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("([Cc]omparison to (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("([Cc]omparison with (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                ("([Cc]omparison among (NP_\w+ ?(, )?(and |or )?)+)", "first")]
        self.__pos_tagger = PerceptronTagger()
        self.__pattern_words = set()
        self.populate_pattern_words()
        self.for_finding_abbreviations = for_finding_abbreviations
        self.prepositions_to_add = ["to","with","for"] if not for_finding_abbreviations else ["to", "with", "for", "in", "and", "on","from"] 
    
    def populate_pattern_words(self):
        for pattern, parser in self.__hearst_patterns:
            for word in re.sub("[^\w]"," ", pattern).split():
                if "_" not in word and len(word) > 1:
                    self.__pattern_words.add(word)
        
    def prepare(self, rawtext):
        sentences = nltk.sent_tokenize(rawtext.strip())
        sentences = [nltk.word_tokenize(sent) for sent in sentences] 
        sentences = [self.__pos_tagger.tag(sent) for sent in sentences] 

        return sentences

    def chunk(self, rawtext):
        sentences = self.prepare(rawtext.strip())

        all_chunks = []
        for sentence in sentences:
            chunks = self.__np_chunker.parse(sentence) 
            all_chunks.append(self.prepare_chunks(chunks))
        return all_chunks

    def prepare_chunks(self, chunks):
        terms = []
        for chunk in chunks:
            label = None
            try: 
                label = chunk.label()
            except:
                pass
            if label is None: 
                token = chunk[0]
                pos = chunk[1]
                if pos in ['.', ':', '-', '_']:
                    continue
                terms.append(token)
            else:
                for a in chunk:
                    if a[0].lower() not in ["a","an","the"]:
                        terms.append("NP_" + a[0])
        return ' '.join(terms)
    
    def is_descriptional(self, word_tag):
        return word_tag in ["JJ", "JJR", "JJS", "NNS", "NN", "NNP", "NNPS", "PDT", "PRP", "PRP$", "VBG", "VBN","DT"]
    
    def add_description_words_to_np(self, sentence):
        new_sentence = ""
        last_word_np = False
        tagged_words = nlp(sentence)
        for i in range(len(tagged_words)-1,-1,-1):

            word = tagged_words[i]
            word_without_tag = word.text
            normalized_word = word_without_tag.replace("NP_","").replace("_", " ")
            tag = word.tag_
            if i > 0 and tagged_words[i-1].text in self.prepositions_to_add:
                tag = self.__pos_tagger.tag([word.text])[0][1] if len(self.__pos_tagger.tag([word.text])) > 0 else tag
            if ("NP_" in word_without_tag or "NN" in word.tag_) and normalized_word not in self.__pattern_words:
                last_word_np = True
                word_to_add = word_without_tag if "NP_" in word_without_tag else "NP_"+word_without_tag
                new_sentence = word_to_add + " " + new_sentence
            else:
                if normalized_word not in self.__pattern_words and self.is_descriptional(tag) and last_word_np:
                    new_sentence = "NP_" + word_without_tag + " " + new_sentence
                    last_word_np = True
                else:
                    last_word_np = False
                    new_sentence = normalized_word + " " + new_sentence
        return new_sentence.strip()
    
    def should_be_concatenate(self, prev_word, next_word):
        if prev_word.replace("_","") in self.__pattern_words:
            return False
        return "_" in prev_word and "_" in next_word
    
    def should_preposition_be_added(self, sentence):
        tagged_words = nlp(sentence)
        new_sentence = ""
        i = 0
        while i < len(tagged_words):
            if i > 0 and  i < len(tagged_words) - 1 and tagged_words[i].text in self.prepositions_to_add and "NP_" in tagged_words[i-1].text and "NP_" in tagged_words[i+1].text:
                new_sentence = new_sentence.strip() + "_" + tagged_words[i].text.strip() + "_" + tagged_words[i+1].text[3:].strip() + " "
                i += 2
            else:
                new_sentence = new_sentence +  tagged_words[i].text.strip() + " "
                i += 1
        return new_sentence.strip()
    
    def replace_np_sequences(self, sentence):
        words = ""
        first_word_in_sequence = False

        sentence = self.add_description_words_to_np(sentence)

        tokenized_words = nltk.word_tokenize(sentence.replace("NP_","_"))
        
        for i in range(len(tokenized_words)):
            word = tokenized_words[i]
            if word[0] == "_":
                if not first_word_in_sequence:
                    word = "NP" + word
                    first_word_in_sequence = True
                    words = words + " " + word
                else:
                    words += word
            elif word == "of":
                if i > 0 and i < len(tokenized_words) -1 and self.should_be_concatenate(tokenized_words[i-1], tokenized_words[i+1]):
                    words= words + "_" + word
                else:
                    words= words + " " + word
                    first_word_in_sequence = False
            else:
                words= words + " " + word
                first_word_in_sequence = False
        
        sentence = self.should_preposition_be_added(words.strip())
        sentence = re.sub("\s+"," ", sentence.replace("_ ", " "))
        return sentence

    def build_dictionary_of_hyphened_words(self, text):
        hyphened_words = {}
        for m in re.finditer("\w+-(\w+-?)*", text):
            match_str = m.group(0)
            hyphened_words[match_str.replace("-","")] = match_str
        return hyphened_words

    def cut_out_geo_names(self, text):
        if re.search("@.*@", text) != None:
            extracted_text = re.search("@.*@", text).group(0).replace("NP_","").replace("_", "")
            if len(GeoText(extracted_text).country_mentions) > 0:
                return []
            else:
                return [a for a in text.split() if a.startswith("NP_")]

    def find_compared_items(self, rawtext):

        compared_items = []
        
        self.hyphened_words = self.build_dictionary_of_hyphened_words(rawtext)
        rawtext = rawtext.replace("-","")
        rawtext = rawtext.replace(":","#")
        rawtext = rawtext.replace("(","@").replace(")","@")
        rawtext = rawtext.replace("+","")
        np_tagged_sentences = self.chunk(rawtext)

        for raw_sentence in np_tagged_sentences:
            sentence = self.replace_np_sequences(raw_sentence)
            for prep in ["of", "to","with","between"]:
                sentence = sentence.replace("_%s_"%prep, " of NP_")

            for idx, (hearst_pattern, parser) in enumerate(self.__hearst_patterns):
                for m in re.finditer(hearst_pattern, sentence):
                    match_str = m.group(0)
                    nps = [a for a in match_str.split() if a.startswith("NP_")]                    
                    if len(nps) == 0:
                        continue
                    
                    for word_exp in nps:
                        compared_items.append(self.clean_hyponym_term(word_exp))
        self.hyphened_words = {}
        return list(filter( None, compared_items))

    def find_hyponyms(self, rawtext):

        hyponyms = []
        try:
            lang_of_raw_text = langdetect.detect(rawtext)
            if lang_of_raw_text != "en":
                return []
        except:
            pass
        
        self.hyphened_words = self.build_dictionary_of_hyphened_words(rawtext)
        rawtext = rawtext.replace("-","")
        rawtext = rawtext.replace(":","#")
        rawtext = rawtext.replace("(","@").replace(")","@")
        rawtext = rawtext.replace("+","")
        np_tagged_sentences = self.chunk(rawtext)

        for raw_sentence in np_tagged_sentences:
            sentence = self.replace_np_sequences(raw_sentence)

            for idx, (hearst_pattern, parser) in enumerate(self.__hearst_patterns):
                for m in re.finditer(hearst_pattern, sentence):
                    match_str = m.group(0)
                    if self.for_finding_abbreviations and idx == 4:
                        nps = self.cut_out_geo_names(match_str) 
                    else:
                        nps = [a for a in match_str.split() if a.startswith("NP_")]
                    
                    if len(nps) == 0:
                        continue
                    
                    if parser == "first":
                        general = nps[0]
                        specifics = nps[1:]
                    else:
                        general = nps[-1]
                        specifics = nps[:-1]

                    for i in range(len(specifics)):
                        specific_cleaned = self.clean_hyponym_term(specifics[i])
                        general_cleaned = self.clean_hyponym_term(general)
                        if specific_cleaned == "intervention":
                            continue
                        if specific_cleaned != "" and general_cleaned != "":
                            hyponyms.append((specific_cleaned, general_cleaned, sentence, idx))
        self.hyphened_words = {}
        return hyponyms
    
    def find_interventions_pairs(self, rawtext):

        hyponyms = []
        rawtext = rawtext.replace("-","_")
        rawtext = rawtext.replace(":","#")
        rawtext = rawtext.replace("(","@").replace(")","@")
        np_tagged_sentences = self.chunk(rawtext)

        for raw_sentence in np_tagged_sentences:
            sentence = self.replace_np_sequences(raw_sentence)
            
            hyponyms.append(([self.clean_hyponym_term(word) for word in sentence.split() if word.startswith("NP_") and self.clean_hyponym_term(word) != ""],sentence))

        return hyponyms

    def clean_expression_with_filtered_words(self, words, filtered_words):
        words_filtered = []
        cleaned = False
        for i in range(len(words)):
            if not cleaned and (words[i][0] in filtered_words or words[i][1] in filtered_words or words[i][2]):
                continue
            else:
                words_filtered.append(words[i])
                cleaned = True
        return words_filtered

    def clean_hyponym_term(self, term):
        cleaned_expression = term.replace("NP_","").replace("_", " ").strip()
        try:
            for word in self.hyphened_words:
                cleaned_expression = cleaned_expression.replace(word, self.hyphened_words[word])
        except:
            pass
        filtered_words = ["use", "related", "current", "various", "improved", "friendly","one", "new", "better","low","output","minimal","case","us","type","study",\
        "particular","appropriate", "different", "good", "bad", "better","scenario"]
        words = []
        splitted_words = nlp(cleaned_expression)
        for word in splitted_words:
            word_without_tag = text_normalizer.normalize_text(word.text)
            if word_without_tag not in text_normalizer.stopwords_all:
                lmtzed_word = lmtzr.lemmatize(word_without_tag)
                if text_normalizer.is_abbreviation(lmtzed_word):
                    words.append((lmtzed_word, word.lemma_, word.is_stop))
                else:
                    if len(lmtzed_word) > 1:
                        words.append((lmtzed_word.lower(), word.lemma_.lower(), word.is_stop))
        filtered_words.extend(text_normalizer.stopwords_all)
        words_filtered = self.clean_expression_with_filtered_words(words, filtered_words)
        words_filtered = list(reversed(self.clean_expression_with_filtered_words(list(reversed(words_filtered)), filtered_words)))
        new_expression = " ".join([word[0] for word in words_filtered])
        return new_expression