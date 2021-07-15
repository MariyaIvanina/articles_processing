import os
import pandas as pd
from text_processing import text_normalizer
import pandas as pd
import nltk
import re
from utilities import excel_reader

class InterventionsSearchForLabeling:

    def __init__(self, filename):
        self.interventions_extension_words = [
            'action',
            'algorithm',
            'approach',
            'concept',
            'definition',
            'developing',
            'development',
            'effort',
            'framework',
            'facility',
            'implementation',
            'initiative',
            'instrument',
            'intervention',
            'infrastructure',
            'measure',
            'method',
            'methodology',
            'model',
            'option',
            'outcome',
            'paradigm',
            'perspective',
            'policy',
            'principle',
            'procedure',
            'program',
            'project',
            'scheme',
            'service',
            'strategy',
            'targeting',
            'technique',
            'theory',
            'tool',
            'programme']
        self.create_interventions_dictionary(filename)

    def create_interventions_dictionary(self, filename):
        self.interventions_set = {}
        interventions_df = excel_reader.ExcelReader().read_df_from_excel(filename)
        for i in range(len(interventions_df)):
            narrow_concept = re.sub(r"\bprogramme\b", "program", interventions_df["Narrow concept"].values[i]).strip()
            abbreviation_meanings = interventions_df["Abbreviation meaning"].values[i].strip()
            middle_layers = [""]
            if "Middle Layer names" in interventions_df.columns:
                middle_layers = [w.strip() for w in interventions_df["Middle Layer names"].values[i].split(";") if w.strip() != ""]
            if narrow_concept not in self.interventions_set:
                self.interventions_set[narrow_concept] = {}
            self.interventions_set[narrow_concept][abbreviation_meanings] = (interventions_df["Intervention class"].values[i],middle_layers)
        self.interventions_for_extension = {}
        last_part_interventions = {}
        for interv in self.interventions_set:
            interv_part = interv.split()
            if len(interv_part) == 1:
                continue
            first_part, last_part = " ".join(interv_part[:-1]), interv_part[-1]
            if first_part not in last_part_interventions:
                last_part_interventions[first_part] = set()
            last_part_interventions[first_part].add(last_part)
        for intervention in last_part_interventions:
            cnt = 0
            for word in last_part_interventions[intervention]:
                if word in self.interventions_extension_words:
                    cnt += 1
            if cnt == len(last_part_interventions[intervention]):
                all_interventions = [intervention + " " + w for w in last_part_interventions[intervention]]
                intervention_name = intervention + " intervention"
                for word in last_part_interventions[intervention]:
                    self.interventions_for_extension[intervention + " " + word] = (intervention, intervention_name, all_interventions)


    def find_interventions_from_dictionary(self, articles_df, search_engine_inverted_index,  _abbreviations_resolver):
        columns_to_label = ["interventions_found","intervention_labels", "interventions_found_raw", 
            "technology intervention", "socioeconomic intervention","ecosystem intervention","storage intervention","mechanisation intervention"]
        for column_name in columns_to_label:
            articles_df[column_name] = ""
        used_interventions = set()
        for key in self.interventions_set:
            for meaning in self.interventions_set[key]:
                intervention_name = key
                keywords_to_find = [key]
                intervention_name_raw = key
                if key in self.interventions_for_extension:
                    if key not in used_interventions:
                        used_interventions = used_interventions.union(set(self.interventions_for_extension[key][2]))
                        intervention_name_raw = self.interventions_for_extension[key][0]
                        intervention_name = self.interventions_for_extension[key][1]
                        keywords_to_find = self.interventions_for_extension[key][2]
                    else:
                        continue
                for article in search_engine_inverted_index.find_articles_with_keywords(
                        keywords_to_find, 0.92, extend_with_abbreviations = True, extend_abbr_meanings = meaning):
                    extended_with_abbreviations_key = _abbreviations_resolver.replace_abbreviations(intervention_name, meaning)
                    if articles_df["intervention_labels"].values[article] == "":
                        articles_df["intervention_labels"].values[article] = set()
                        articles_df["interventions_found_raw"].values[article] = set()
                        articles_df["interventions_found"].values[article] = set()
                    articles_df["intervention_labels"].values[article].add(self.interventions_set[key][meaning][0])
                    articles_df["interventions_found_raw"].values[article].add(intervention_name_raw)
                    for middle_layer in self.interventions_set[key][meaning][1]:
                        interv_full_name = extended_with_abbreviations_key if middle_layer == "" else (middle_layer + "/" + extended_with_abbreviations_key)
                        articles_df["interventions_found"].values[article].add(self.interventions_set[key][meaning][0] + "/" + interv_full_name)
                        if middle_layer != "" and middle_layer != "other":
                            articles_df["interventions_found"].values[article].add(self.interventions_set[key][meaning][0] + "/" + middle_layer + "/")
                        column_name = self.interventions_set[key][meaning][0].lower()
                        if articles_df[column_name].values[article] == "":
                            articles_df[column_name].values[article] = set()
                        articles_df[column_name].values[article].add(interv_full_name)
                        if middle_layer != "" and middle_layer != "other":
                            articles_df[column_name].values[article].add(middle_layer + "/")

        for column_name in columns_to_label: 
            articles_df = text_normalizer.replace_string_default_values(articles_df, column_name)
        return articles_df

