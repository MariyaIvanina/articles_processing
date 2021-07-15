from utilities import excel_reader
from text_processing import text_normalizer

class ColumnDataRenamer:

    def __init__(self, values_dict={},
            dict_filename="", keyword_column = "Keyword",
            high_level_label_column = "Renamed keyword", status_logger = None):
        self.keyword_dictionary = {}
        if dict_filename.strip():
            self.load_dictionary(dict_filename, keyword_column, high_level_label_column)
        else:
            for key in values_dict:
                if key not in self.keyword_dictionary:
                    self.keyword_dictionary[key] = []
                if type(values_dict[key]) == list:
                    self.keyword_dictionary[key].extend(values_dict[key])
                else:
                    self.keyword_dictionary[key].append(values_dict[key])
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
            if dict_df[keyword_column].values[i] not in self.keyword_dictionary:
                self.keyword_dictionary[dict_df[keyword_column].values[i]] = []
            self.keyword_dictionary[dict_df[keyword_column].values[i]].append(dict_df[high_level_label_column].values[i])

    def label_articles_with_renamed_values(self, column_to_rename, articles_df,
            column_to_save="",
            search_engine_inverted_index=None, _abbreviations_resolver=None,
            print_all = False):
        found_words_to_rename = {}
        should_be_list_values = False
        all_renamed = 0
        if column_to_save.strip():
            articles_df[column_to_save] = ""
        else:
            column_to_save = column_to_rename

        for i in range(len(articles_df)):

            if i % 100 == 0:
                self.log_percents(i/len(articles_df)*90)

            if type(articles_df[column_to_rename].values[i]) == list:
                should_be_list_values = True
                new_values = []
                for val in articles_df[column_to_rename].values[i]:
                    if val in self.keyword_dictionary:
                        if val not in found_words_to_rename:
                            found_words_to_rename[val] = 0
                        found_words_to_rename[val] += 1
                        all_renamed += 1
                        new_values.extend(self.keyword_dictionary[val])
                    else:
                        new_values.append(val)
                articles_df[column_to_save].values[i] = list(set(new_values))
            else:
                if articles_df[column_to_rename].values[i] in self.keyword_dictionary:
                    all_renamed += 1
                    if articles_df[column_to_rename].values[i] not in found_words_to_rename:
                        found_words_to_rename[articles_df[column_to_rename].values[i]] = 0
                    found_words_to_rename[articles_df[column_to_rename].values[i]] += 1
                    articles_df[column_to_save].values[i] = self.keyword_dictionary[articles_df[column_to_rename].values[i]][0]
                else:
                    articles_df[column_to_save].values[i] = articles_df[column_to_rename].values[i]

        if should_be_list_values:
            articles_df = text_normalizer.replace_string_default_values(articles_df, column_to_save)
        if print_all:
            for key in found_words_to_rename:
                print(key, found_words_to_rename[key])
        print("Articles with renamed: %d"%all_renamed)
        return articles_df

