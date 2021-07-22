import time
from text_processing import text_normalizer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd
import spacy
from text_processing import geo_names_finder
from utilities import excel_writer
from utilities import excel_reader
import os
import textdistance

lmtzr = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

class AuthorAffiliationExtractor:

    def __init__(self):
        self.universities = {}
        self.processed_by_google = {}
        self.mapping_words = {}
        self.univ_mention_counts = {}
        self.universities_by_city = {}
        self.universities_by_country = {}
        self.universities_by_first_letters = {}
        self.geo_names_finder = geo_names_finder.GeoNameFinder()
        self.file_with_affiliations = '../data/map_affiliations.xlsx'
        self.need_to_remap = False
        self.initialize_author_affiliation()

    def initialize_author_affiliation(self):
        self.load_map_affiliations()
        self.load_map_clipped_words()
        
        print("Length of map of affiliations: %d" % len(self.universities))
        print("Count of unique normalized affiliations: %d" % self.count_unique_normalized_universities())

    def load_map_affiliations(self):
        if os.path.exists(self.file_with_affiliations):
            map_df = excel_reader.ExcelReader().read_df_from_excel(self.file_with_affiliations)
            for i in range(len(map_df)):
                self.universities[map_df["raw_university_name"].values[i]] = (map_df["normalized_name"].values[i], map_df["city"].values[i], map_df["country"].values[i])
                self.add_university_to_dict(map_df["normalized_name"].values[i], map_df["raw_university_name"].values[i], map_df["city"].values[i], map_df["country"].values[i])
                self.univ_mention_counts[map_df["normalized_name"].values[i]] = map_df["count_mentions"].values[i]

    def load_map_clipped_words(self):
        map_df = excel_reader.ExcelReader().read_df_from_excel('../data/map_clipped_words.xlsx')
        for i in range(len(map_df)):
            self.mapping_words[map_df["word"].values[i]] = map_df["mapping_word"].values[i]

    def search_institution(self, query):
        r = ""
        secret_key = os.getenv("GOOGLE_SECRET_KEY", "")
        while r == "":
            try:
                r = requests.get(
                'https://kgsearch.googleapis.com/v1/entities:search?query=%s&key=%s&prefix=True' % (query, secret_key)
                )
                break
            except:
                print("Connection refused by the server..")
                time.sleep(5)
                continue
        
        obj = json.loads(r.text)
        obj_name = ""
        if 'itemListElement' in obj and len(obj['itemListElement']) > 0:
            obj = obj['itemListElement'][0]['result']
            if 'name' in obj:
                obj_name = obj['name']
                self.processed_by_google[query] = self.normalize_answer(obj_name)
        return self.normalize_answer(obj_name)

    def get_mapping_for_word(self, word):
        if word in self.mapping_words:
            return self.mapping_words[word]
        return word

    def normalize_answer(self, obj_name):
        obj_name = re.sub('\s+', ' ', re.sub(r"[^\w\s']", " ",obj_name.replace(".","").replace("ᅟ","").lower()).strip())
        words_res = []
        for word in obj_name.split():
            if word not in text_normalizer.stopwords_all and len(word) > 1 and word != "part":
                lemma_word = lmtzr.lemmatize(self.get_mapping_for_word(word))
                if len(lemma_word) > 1:
                    words_res.append(lemma_word)
        return " ".join(words_res)

    def contains_digits(self, text):
        return re.match("\d+", text) != None

    def get_full_raw_university_name(self, raw_university_name, city, country):
        if city != "":
            raw_university_name = raw_university_name + " " + city
        if country != "":
            raw_university_name = raw_university_name + " " + country
        return raw_university_name

    def find_common_university_name(self, raw_university_name, city, country):
        if raw_university_name.strip() == '':
            return ''
        new_raw_university_name = self.get_full_raw_university_name(self.normalize_answer(raw_university_name), city, country)
        if new_raw_university_name in self.universities:
            return self.universities[new_raw_university_name][0]
        self.need_to_remap = True
        self.universities[new_raw_university_name] = (new_raw_university_name,city,country)
        return self.universities[new_raw_university_name][0]

    def get_institution_name(self, affiliation_parts):
        univ_words = ["univ", "inst", "college", "academy", "r&d", "foundation", "research center", "hochschul", \
                      "centre", "center", "organis", "dept", "depart", "school", "research", "division", "program", "project"]
        for word in univ_words:
            for affil_part in reversed(affiliation_parts):
                if word in affil_part:
                    return text_normalizer.remove_accented_chars(affil_part)
        return " ".join([text_normalizer.remove_accented_chars(affil_part) for affil_part in affiliation_parts])

    def save_map_affiliations(self):
        affiliation_df_list = []
        for key, val in self.universities.items():
            affiliation_df_list.append((key,val[0],val[1],val[2], self.univ_mention_counts[val[0]]))
        excel_writer.ExcelWriter().save_data_in_excel(affiliation_df_list, ["raw_university_name", "normalized_name", "city", "country", "count_mentions"], self.file_with_affiliations, width_column=70)
        
    def count_unique_normalized_universities(self):
        set_afffiliations = set()
        for key, val in self.universities.items():
            set_afffiliations.add(self.universities[key])
        return len(set_afffiliations)

    def add_university_to_dict(self, univ_name, raw_name, city_name, country):
        if city_name not in self.universities_by_city:
            self.universities_by_city[city_name] = set()
        self.universities_by_city[city_name].add((univ_name, country, raw_name))
        if univ_name not in self.univ_mention_counts:
            self.univ_mention_counts[univ_name] = 0
        self.univ_mention_counts[univ_name] += 1

    def extract_country_name(self, raw_university_name, last_part):
        if self.geo_names_finder.contains_usa_name(raw_university_name) or re.search(r"\bUS\b", raw_university_name) != None:
            return "united states"
        if self.geo_names_finder.contains_uk_name(raw_university_name):
            return "united kingdom"
        if "russia" in raw_university_name.lower():
            return "russian federation"
        if last_part.lower() in self.geo_names_finder.countries_map:
            return last_part.lower()
        return self.geo_names_finder.get_country_from_text_with_and_without_accented_chars(raw_university_name)

    def normalize_affiliations(self, affiliation_str):
        raw_university_name = re.compile("\[.*?\]").sub("",affiliation_str).split(';')
        raw_university_names = []
        for a in raw_university_name:
            raw_university_names.append([affil_part.replace('.','').replace('\"','').replace('–', ' ').replace('-', ' ').replace('—', ' ').replace('_', ' ').strip().lower() for affil_part in a.split(',')])
        affiliations = []
        pattern = "(p ?o box (\w*)? \d+)|(p ?o box \d+)|(p ?o box)|\(.*?\)|(/.*)|peoples r|of|&|ox1 3ps|\d+|[Cc]ampus"
        for affiliation in raw_university_names:
            affiliations.append( [self.normalize_answer(re.compile(pattern).sub("", affil_part).strip()) for affil_part in affiliation])
        university_names = []
        for i in range(len(affiliations)):
            affiliation = affiliations[i]
            raw_university_name_parts = []
            for affil_part in affiliation:
                if len(affil_part) > 0:
                    uni_name = " ".join([af for af in affil_part.split() if len(af) > 1])
                    raw_university_name_parts.append(uni_name)
            raw_univ_name = self.get_institution_name(raw_university_name_parts)
            country_name = self.extract_country_name(raw_university_name[i], "" if len(raw_university_name_parts) == 0 else raw_university_name_parts[-1])
            city_name = self.geo_names_finder.get_city_from_text_with_and_without_accented_chars(raw_university_name[i])
            university_names.append((raw_univ_name, city_name, country_name))
        return university_names

    def deduplicate_words_in_affiliations(self, affiliation):
        words = set()
        res = []
        for word in reversed(affiliation.split()):
            word = word.strip()
            if word not in words:
                words.add(word)
                res.append(word)
        return " ".join(reversed(res))

    def replace_geo_points_in_name(self, affil, without_country = False):
        new_affil = affil[0]
        if affil[1] != "":
            new_affil = new_affil.replace(affil[1], " ")
        if not without_country and affil[2] != "":
            new_affil = new_affil.replace(affil[2], " ")
        return new_affil

    def sorted_string(self, word_exp):
    	return " ".join(sorted(word_exp.split()))

    def are_affiliations_similar(self, first_affil, second_affil, without_country = False, threshold = 0.88):
        old_first_affil = first_affil[0]
        old_second_affil = second_affil[0]
        first_affil = self.replace_geo_points_in_name(first_affil,without_country)
        second_affil = self.replace_geo_points_in_name(second_affil,without_country)
        filtered_words = ["university", "institute", "research", "center", "centre", "department", "college"]
        for filter_word in filtered_words:
            first_affil = first_affil.replace(filter_word, " ")
            second_affil = second_affil.replace(filter_word, " ")
        first_affil = re.sub("\s+", " ", first_affil).strip()
        second_affil = re.sub("\s+", " ", second_affil).strip()
        return textdistance.levenshtein.normalized_similarity(first_affil, second_affil) >= threshold or\
                textdistance.levenshtein.normalized_similarity(old_first_affil, old_second_affil) >= threshold or\
                textdistance.levenshtein.normalized_similarity(self.sorted_string(first_affil), self.sorted_string(second_affil)) >= threshold\
                or textdistance.levenshtein.normalized_similarity(
                    self.sorted_string(old_first_affil), self.sorted_string(old_second_affil)) >= threshold 

    def get_major_country(self, country_for_city):
        max_country,max_country_count = "",0
        for country in country_for_city:
            if country_for_city[country] > max_country_count and country != "":
                max_country = country
                max_country_count = country_for_city[country]
        return max_country
    
    def remap_by_city(self, threshold=0.88):
        for city in self.universities_by_city:
            new_mapping = {}
            if city == "":
                continue
            country_for_city = {}
            list_of_words = [(expr, self.univ_mention_counts[expr[0]]) for expr in self.universities_by_city[city]]
            list_of_words = sorted(list_of_words,key = lambda x: x[1],reverse=True)
            for i in range(len(list_of_words)):
                for j in range(i+1, len(list_of_words)):
                    if textdistance.levenshtein.normalized_similarity(
                            list_of_words[i][0][0], list_of_words[j][0][0]) >= threshold and list_of_words[j][0][0] not in new_mapping:
                        new_mapping[list_of_words[j][0][0]] = list_of_words[i][0][0] if list_of_words[i][0][0] not in new_mapping else new_mapping[list_of_words[i][0][0]]
                country_name = list_of_words[i][0][1] 
                if country_name not in country_for_city:
                    country_for_city[country_name] = 0
                country_for_city[country_name] += self.univ_mention_counts[list_of_words[i][0][0]]
                if list_of_words[i][0][0] not in new_mapping:
                    new_mapping[list_of_words[i][0][0]] = list_of_words[i][0][0]

            major_country = self.get_major_country(country_for_city)
            
            new_set = set()
            for expr in self.universities_by_city[city]:
                new_set.add((new_mapping[expr[0]], major_country if expr[1] == "" else expr[1], expr[2]))
            self.universities_by_city[city] = new_set

    def fill_universities_by_country(self):
        for city in self.universities_by_city:
            for univ in self.universities_by_city[city]:
                if univ[1] not in self.universities_by_country:
                    self.universities_by_country[univ[1]] = set()
                self.universities_by_country[univ[1]].add((univ[0], city, univ[2]))

    def remap_by_country(self, threshold = 0.88):
        self.fill_universities_by_country()
        for country in self.universities_by_country:
            if country == "":
                continue
            new_mapping = {}
            list_of_words = [(expr, self.univ_mention_counts[expr[0]] if expr[1] != "" else -1) for expr in self.universities_by_country[country]]
            list_of_words = sorted(list_of_words,key = lambda x: x[1],reverse=True)
            for i in range(len(list_of_words)):
                i_map_val = (list_of_words[i][0][0], list_of_words[i][0][1])
                for j in range(i+1, len(list_of_words)):
                    new_map_val = (list_of_words[j][0][0], list_of_words[j][0][1])
                    if self.are_affiliations_similar(list_of_words[i][0], list_of_words[j][0], True, threshold) and (list_of_words[j][0][1] == "" or  list_of_words[i][0][1] == list_of_words[j][0][1]):
                        if new_map_val not in new_mapping:
                            new_mapping[new_map_val] = i_map_val if i_map_val not in new_mapping else new_mapping[i_map_val]
                if i_map_val not in new_mapping:
                    new_mapping[i_map_val] = i_map_val
            new_set = set()
            for expr in self.universities_by_country[country]:
                new_map_val = new_mapping[(expr[0], expr[1])]
                new_set.add((new_map_val[0], new_map_val[1], expr[2]))
            self.universities_by_country[country] = new_set

    def fill_dictionary_with_new_mappings(self):
        for first_letters in self.universities_by_first_letters:
            for univ in self.universities_by_first_letters[first_letters]:
                self.universities[univ[3]] = (self.deduplicate_words_in_affiliations(self.get_full_raw_university_name(univ[0], univ[1], univ[2])), univ[1], univ[2])
                self.univ_mention_counts[self.universities[univ[3]][0]] = self.univ_mention_counts[univ[0]]

    def fill_universities_by_first_letters(self):
        for country in self.universities_by_country:
            for univ in self.universities_by_country[country]:
                if univ[0][:2] not in self.universities_by_first_letters:
                    self.universities_by_first_letters[univ[0][:2]] = set()
                self.universities_by_first_letters[univ[0][:2]].add((univ[0], univ[1], country, univ[2]))

    def get_weight(self, univ):
        weight = 0
        if univ[1] == "":
            weight -= 1
        if univ[2] == "":
            weight -= 1
        if weight == 0:
            return self.univ_mention_counts[univ[0]]
        return weight

    def is_geo_position_similar(self, first_univ, second_univ):
        if first_univ[1] == second_univ[1] and first_univ[2] == second_univ[2]:
            return True
        if first_univ[2] == second_univ[2] and second_univ[1] == "":
            return True 
        if second_univ[1] == "" and second_univ[2] == "":
            return True
        return False

    def remap_by_first_letters(self, threshold=0.88):
        self.fill_universities_by_first_letters()
        for first_letters in self.universities_by_first_letters:
            new_mapping = {}
            list_of_words = [(expr, self.get_weight(expr)) for expr in self.universities_by_first_letters[first_letters]]
            list_of_words = sorted(list_of_words,key = lambda x: x[1],reverse=True)
            for i in range(len(list_of_words)):
                i_map_val = (list_of_words[i][0][0], list_of_words[i][0][1], list_of_words[i][0][2])
                for j in range(i+1, len(list_of_words)):
                    new_map_val = (list_of_words[j][0][0], list_of_words[j][0][1], list_of_words[j][0][2])
                    if self.are_affiliations_similar(list_of_words[i][0], list_of_words[j][0], threshold) and self.is_geo_position_similar(list_of_words[i][0], list_of_words[j][0]):
                        if new_map_val not in new_mapping:
                            new_mapping[new_map_val] = i_map_val if i_map_val not in new_mapping else new_mapping[i_map_val]
                if i_map_val not in new_mapping:
                    new_mapping[i_map_val] = i_map_val
            new_set = set()
            for expr in self.universities_by_first_letters[first_letters]:
                new_map_val = new_mapping[(expr[0], expr[1], expr[2])]
                new_set.add((new_map_val[0], new_map_val[1], new_map_val[2], expr[3]))
            self.universities_by_first_letters[first_letters] = new_set

    def start_affiliation_remapping(self, raw_affiliations):
        t0 = time.time()
        print("started affiliation processing")
        for i in range(len(raw_affiliations)):
            if i % 3000 == 0 or i == len(raw_affiliations) -1:
                print("Process %d articles" %i)
            affiliations_names = self.normalize_affiliations(raw_affiliations[i])
            for univ_name, city_name, country_name in affiliations_names:
                if univ_name != "":
                    raw_univ_name = self.find_common_university_name(univ_name, city_name, country_name)
                    self.add_university_to_dict(univ_name, raw_univ_name, city_name, country_name)
        if self.need_to_remap:
            print("Count of unique normalized affiliations: %d" % self.count_unique_normalized_universities())
            self.remap_by_city(0.88)
            self.remap_by_country(0.9)
            self.remap_by_first_letters(0.9)
            self.fill_dictionary_with_new_mappings()
            
            print("Count of unique normalized affiliations: %d" % self.count_unique_normalized_universities())
            self.save_map_affiliations()
            self.need_to_remap = False

        print("done in %0.3fs." % (time.time() - t0))

    def find_university_affiliations_from_string(self, raw_affiliation):
        university_names = []
        affiliations_names = self.normalize_affiliations(raw_affiliation)
        for raw_univ_name,city_name, country_name in affiliations_names:
            univ_name = raw_univ_name
            if raw_univ_name != "":
                univ_name = self.find_common_university_name(raw_univ_name, city_name, country_name)
            university_names.append(univ_name)
        return list(set([univ_name for univ_name in university_names if univ_name != '']))