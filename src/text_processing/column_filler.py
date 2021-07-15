from utilities import excel_reader
from text_processing import text_normalizer

class ColumnFiller:

    def __init__(self, dict_filename, keyword_column = "Keyword", high_level_label_column = "High level label", status_logger = None):
        self.load_dictionary(dict_filename, keyword_column, high_level_label_column)
        self.status_logger = status_logger

    def log_percents(self, percent):
        if self.status_logger is not None:
            self.status_logger.update_step_percent(percent)

    def load_dictionary(self, dict_filename, keyword_column, high_level_label_column):
        dict_df = excel_reader.ExcelReader().read_file(dict_filename)
        assert keyword_column in dict_df.columns or high_level_label_column in dict_df.columns,\
         "Check column names in dictionary, %s and %s are not found"%(keyword_column, high_level_label_column)
        self.keyword_dictionary = {}
        for i in range(len(dict_df)):
            if dict_df[high_level_label_column].values[i] not in self.keyword_dictionary:
                self.keyword_dictionary[dict_df[high_level_label_column].values[i]] = []
            self.keyword_dictionary[dict_df[high_level_label_column].values[i]].append(dict_df[keyword_column].values[i])

    def label_articles_with_outcomes(self, label_column, articles_df, 
            search_engine_inverted_index, _abbreviations_resolver, label_details_column = "", 
            threshold=0.92, print_all = False, keep_hierarchy=False, resolve_abbreviations=False):
        found_high_level_values = {}
        articles_df[label_column] = ""
        if label_details_column != "":
            articles_df[label_details_column] =""
        for idx, key_1 in enumerate(self.keyword_dictionary):
            print("High level label: %s"%key_1)
            if idx % 3 == 0:
                self.log_percents(idx/len(self.keyword_dictionary)*90)
            for key in self.keyword_dictionary[key_1]:
                for article_index in search_engine_inverted_index.find_articles_with_keywords(
                        [key], threshold = threshold, extend_with_abbreviations = True):
                    if articles_df[label_column].values[article_index] == "":
                        articles_df[label_column].values[article_index] = set()
                    key_resolved = key_1
                    if resolve_abbreviations:
                        key_resolved = _abbreviations_resolver.replace_abbreviations(key_1)
                    articles_df[label_column].values[article_index].add(key_resolved)
                    if label_details_column != "":
                        if articles_df[label_details_column].values[article_index] == "":
                            articles_df[label_details_column].values[article_index] = set()
                        full_keyword_name = key
                        if resolve_abbreviations:
                            full_keyword_name = _abbreviations_resolver.replace_abbreviations(key)
                        full_keyword_path_name = "%s/%s"%(key_1, full_keyword_name)
                        if keep_hierarchy:
                            articles_df[label_details_column].values[article_index].add(full_keyword_path_name)
                            articles_df[label_details_column].values[article_index].add(key_1 + "/")
                        else:
                            articles_df[label_details_column].values[article_index].add(full_keyword_name)
                    if key not in found_high_level_values:
                        found_high_level_values[key] = set()
                    found_high_level_values[key].add(article_index)
        total_outcomes = 0
        for i in range(len(articles_df)):
            if articles_df[label_column].values[i] != "":
                total_outcomes += 1
        articles_df = text_normalizer.replace_string_default_values(articles_df, label_column)
        if label_details_column != "":
            articles_df = text_normalizer.replace_string_default_values(articles_df, label_details_column)
        if print_all:
            for key in found_high_level_values:
                print(key, len(found_high_level_values[key]))
        print("Labelled articles with outcomes: %d"%total_outcomes)
        return articles_df

