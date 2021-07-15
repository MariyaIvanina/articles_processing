import nltk
from allennlp.predictors.predictor import Predictor
from text_processing import text_normalizer
import re
import spacy
from interventions_labeling_lib import hearst_pattern_finder
from interventions_labeling_lib import hyponym_statistics
import os
import pickle
from text_processing import concepts_merger

nlp = spacy.load('en_core_web_sm')

class ComparedTermsFinder:

    def __init__(self,search_engine_inverted_index, abbreviation_resolver, folder_to_save_temp_res):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.filter_words = ["compare","compares","compared","comparing"]
        self.hyp_stat = hyponym_statistics.HyponymStatistics({},search_engine_inverted_index, abbreviation_resolver,{},{})
        self.hearst_pattern = hearst_pattern_finder.HearstPatterns(for_finding_comparison = True)
        self.folder_to_save_temp_res = folder_to_save_temp_res

    def find_compared_terms_in_sentence(self, parse_sentence):
        for pair in parse_sentence:
            if text_normalizer.contain_verb_form(pair["verb"].lower(), self.filter_words):
                first_part = ""
                second_part = ""
                verb = ""
                description = pair["description"]
                for m in re.finditer("\[(ARG|V).*?:.*?\]", description):
                    if "ARG2" in m.group(0):
                        second_part = m.group(0).split(":")[1][:-1].strip()
                    if "ARG1" in m.group(0):
                        first_part = m.group(0).split(":")[1][:-1].strip()
                return first_part, second_part, pair["verb"], pair["description"]
        return "", "","",""

    def break_phrase_into_parts(self, text):
        parts = text.split("and")
        first_part, second_part = "", ""
        if len(parts[0].split()) == 1 and len(parts[1].split()) > 1:
            first_part = parts[0].strip() + " " + " ".join(parts[1].split()[1:]).strip()
            second_part = parts[1].strip()
        elif len(parts[1].split()) == 1:
            first_part = parts[0].strip()
            second_part = parts[1].strip()
        return first_part, second_part

    def clean_pattern_words(self, text):
        return self.hyp_stat.clean_concept(self.hearst_pattern.clean_hyponym_term(text.replace("NP_","").replace("_"," ")))

    def clean_result(self, text, search_engine_inverted_index):
        cleaned_text = self.hearst_pattern.clean_hyponym_term(text)
        if len(cleaned_text.split()) < 4:
            return  self.hyp_stat.clean_concept(cleaned_text)
        np_text = self.hearst_pattern.replace_np_sequences(text).replace("_of_", " of NP_")
        for phr in nlp(np_text).noun_chunks:
            cleaned_text =  self.clean_pattern_words(phr.text)
            if cleaned_text == "":
                continue
            freq_percent = len(search_engine_inverted_index.find_articles_with_keywords([cleaned_text],threshold=1.0, extend_with_abbreviations=False))/search_engine_inverted_index.total_articles_number
            if len(set([tag.text for tag in phr.rights]).intersection(set(["of", "in"]))) > 0 and freq_percent  > 0.01:
                continue
            if freq_percent < 0.01:
                return cleaned_text
        for phr in nlp(np_text).noun_chunks:
            cleaned_text = self.clean_pattern_words(phr.text)
            if cleaned_text != "":
                return cleaned_text
        return cleaned_text

    def find_raw_compared_items_sentence_parsing(self, articles_df, search_engine_inverted_index):
        res_all = {}
        
        all_articles =  search_engine_inverted_index.find_articles_with_keywords(self.filter_words, threshold = 1.0, extend_with_abbreviations = False)
        cnt = 0
        for i in all_articles:
            try:
                if i not in res_all:
                    res_all[i] = []
                cnt += 1
                if cnt % 500 == 0:
                    print("%d artciles processed"%cnt)
                sentences = [text_normalizer.remove_accented_chars(articles_df["title"].values[i].lower()\
                    if articles_df["title"].values[i].isupper() else articles_df["title"].values[i])]
                sentences.extend(nltk.sent_tokenize(text_normalizer.remove_accented_chars(articles_df["abstract"].values[i])))
                for sentence in sentences:
                    if text_normalizer.has_word_in_sentence(sentence, self.filter_words):
                        parse_sentence = self.predictor.predict(sentence=sentence)["verbs"]
                        first_part, second_part, verb, description = self.find_compared_terms_in_sentence(parse_sentence)

                        for v in parse_sentence:
                            res = {}
                            for m in re.finditer("\[(ARG|V).*?:.*?\]", v["description"]):
                                tag = m.group(0).split(":")[0][1:].strip()
                                tag_text = m.group(0).split(":")[1][:-1].strip()
                                if tag not in res:
                                    res[tag] = []
                                res[tag].append(tag_text)
                            for tag in res:
                                for arg in res[tag]:
                                    if verb in arg and tag != "V" and (first_part == "" or first_part not in arg or (first_part in arg and nlp(arg.split()[0])[0].tag_ == "IN")):
                                        search_tag = "ARG1"
                                        if tag == "ARGV-TMP":
                                            search_tag = "ARG0"
                                        if re.search("ARG\d+", tag):
                                            search_tag = "ARG" + str(int(re.search("\d+", tag).group(0))-1)
                                        if search_tag in res:
                                            first_part = res[search_tag][0]
                                            break
                        if first_part == "":
                            for m in re.finditer("\[(ARG|V).*?:.*?\]", description):
                                if "ARG1" in m.group(0):
                                    first_part = m.group(0).split(":")[1][:-1].strip()
                                    break
                        if (" and " in first_part and second_part == "") or (" and " in second_part and first_part == ""):
                            first_part, second_part = self.break_phrase_into_parts(first_part + second_part)
                        res_all[i].append((sentence, first_part, second_part))
            except KeyboardInterrupt:
                raise
            except Exception as err:
                print("error occured for %d article"%i)
                print(err)
        return res_all

    def clean_found_compared_items(self, res_all, search_engine_inverted_index):
        res_all_cleaned = {}
        for i in res_all:
            res_all_cleaned[i] = []
            for part in res_all[i]:
                res_all_cleaned[i].append((self.clean_result(part[1], search_engine_inverted_index).strip(), self.clean_result(part[2], search_engine_inverted_index).strip()))
        return res_all_cleaned

    def fill_compared_items(self, articles_df, search_engine_inverted_index):
        if not os.path.exists(self.folder_to_save_temp_res):
            os.makedirs(self.folder_to_save_temp_res)
        
        if os.path.exists(os.path.join(self.folder_to_save_temp_res, "res_all.pickle")):
            res_all = pickle.load(open(os.path.join(self.folder_to_save_temp_res, "res_all.pickle"), "rb"))
        else:
            res_all = self.find_raw_compared_items_sentence_parsing(articles_df, search_engine_inverted_index)
            pickle.dump(res_all, open(os.path.join(self.folder_to_save_temp_res, "res_all.pickle"),"wb"))
        
        if os.path.exists(os.path.join(self.folder_to_save_temp_res, "res_all_cleaned.pickle")):
            res_all_cleaned = pickle.load(open(os.path.join(self.folder_to_save_temp_res, "res_all_cleaned.pickle"), "rb"))
        else:
            res_all_cleaned = self.clean_found_compared_items(res_all, search_engine_inverted_index)
            pickle.dump(res_all_cleaned, open(os.path.join(self.folder_to_save_temp_res, "res_all_cleaned.pickle"),"wb"))

        if os.path.exists(os.path.join(self.folder_to_save_temp_res, "res_all_patterns.pickle")):
            res_all_patterns = pickle.load(open(os.path.join(self.folder_to_save_temp_res, "res_all_patterns.pickle"),"rb"))
        else:
            res_all_patterns = self.find_compared_items_via_patterns(articles_df, search_engine_inverted_index)
            pickle.dump(res_all_patterns, open(os.path.join(self.folder_to_save_temp_res, "res_all_patterns.pickle"),"wb"))

        common_res = self.merge_results(res_all_patterns, res_all_cleaned, search_engine_inverted_index)

        articles_df["compared_terms"] = ""
        for i in range(len(articles_df)):
            articles_df["compared_terms"].values[i] = []
            if i in res_all_cleaned:
                for term in res_all_cleaned[i]:
                    articles_df["compared_terms"].values[i].extend([t.strip() for t in term if t.strip() != ""])
            if i in res_all_patterns:
                for term in res_all_patterns[i]:
                    if term.strip() != "":
                        articles_df["compared_terms"].values[i].append(term.strip())
        return articles_df

    def merge_results(self, res_all_patterns, res_all_cleaned, search_engine_inverted_index):
        common_res = {}
        for i in res_all_cleaned:
            for term in res_all_cleaned[i]:
                if i not in common_res:
                    common_res[i] = []
                common_res[i].extend([t.strip() for t in term if t.strip() != ""])
        for i in res_all_patterns:
            for term in res_all_patterns[i]:
                if i not in common_res:
                    common_res[i] = []
                if term.strip() != "":
                    common_res[i].append(term.strip())
        _concepts_merger = concepts_merger.ConceptsMerger(5)
        for i in common_res:
            for term in common_res[i]:
                _concepts_merger.add_item_to_dict(term, i)
        _concepts_merger.merge_concepts(search_engine_inverted_index)
        for i in common_res:
            new_list = []
            for term in common_res[i]:
                new_list.append(_concepts_merger.new_mapping[term] if term in _concepts_merger.new_mapping else term)
            common_res[i] = new_list
        return common_res

    def find_compared_items_via_patterns(self, articles_df, search_engine_inverted_index):
        res_all = {}
        cnt = 0
        for i in search_engine_inverted_index.find_articles_with_keywords(["comparison"], threshold = 1.0, extend_with_abbreviations = False):
            cnt += 1
            if cnt %500 == 0:
                print("%d articles processed"%cnt)
            if i not in res_all:
                title = text_normalizer.remove_accented_chars(articles_df["title"].values[i].lower()\
                    if articles_df["title"].values[i].isupper() else articles_df["title"].values[i])
                abstract = text_normalizer.remove_accented_chars(articles_df["abstract"].values[i])
                res_all[i] = [self.hyp_stat.clean_concept(expr) for expr in self.hearst_pattern.find_compared_items(title + " . " + abstract)]
        return res_all


