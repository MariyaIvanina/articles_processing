import re
import gensim
import pandas as pd
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_processing import text_normalizer
from utilities import excel_writer, utils
import os
from time import time

class DuplicateFinder:

    def __init__(self, duplicate_info_folder, title_column = "title", abstract_column = "abstract", year_column = "year", dataset_column = "dataset"):
        self.folder = duplicate_info_folder
        self.title_column = title_column
        self.abstract_column = abstract_column
        self.year_column = year_column
        self.dataset_column = dataset_column

    def append_duplicates_into_file(self, df, filename):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        filename = os.path.join(self.folder, filename.replace(".xlsx", "_%d.xlsx"%time()))
        old_df = pd.DataFrame()
        if os.path.exists(filename):
            old_df = pd.read_excel(filename).fillna("")
        old_df = pd.concat([old_df, df], sort=False)
        excelWriter = excel_writer.ExcelWriter()
        excelWriter.save_df_in_excel(old_df, filename)

    def get_tf_idf_matrix(self, article_titles):
        start_time = time()
        titles = [" ".join(text_normalizer.get_normalized_text_with_numbers(title.lower())) for title in article_titles]
        tfidf_vectorizer = TfidfVectorizer(token_pattern="[^ ]+")
        tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
        print(time() - start_time)
        return tfidf_matrix

    def get_duplicate_articles_by_title(self, first_set, second_set, tfidf_matrix, tfidf_matrix_abstract, df1, df2):
        duplicates_pairs= set()
        for first_doc in first_set:
            for second_doc in second_set:
                second_doc_id = second_doc - len(df1)
                first_title = " ".join(text_normalizer.get_normalized_text_with_numbers(df1[self.title_column].values[first_doc]))
                second_title = " ".join(text_normalizer.get_normalized_text_with_numbers(df2[self.title_column].values[second_doc_id]))
                if (utils.normalized_levenshtein_score(
                        first_title, second_title) >= 0.9 and (cosine_similarity(tfidf_matrix_abstract[first_doc], tfidf_matrix_abstract[second_doc]) >= 0.85 or df1[self.abstract_column].values[first_doc].strip() == "" or df2[self.abstract_column].values[second_doc_id].strip() == "")) or\
                 (cosine_similarity(tfidf_matrix[first_doc], tfidf_matrix[second_doc]) >= 0.95 and \
                (cosine_similarity(tfidf_matrix_abstract[first_doc], tfidf_matrix_abstract[second_doc]) >= 0.85 or df1[self.abstract_column].values[first_doc].strip() == "" or df2[self.abstract_column].values[second_doc_id].strip() == "")) or\
                (cosine_similarity(tfidf_matrix[first_doc], tfidf_matrix[second_doc]) > 0.8 and \
                cosine_similarity(tfidf_matrix_abstract[first_doc], tfidf_matrix_abstract[second_doc]) >= 0.94) :
                    duplicates_pairs.add((first_doc, second_doc))
        return duplicates_pairs

    def get_df_for_duplicates(self, df, duplicate_pairs, index_to_delete):
        index_to_leave = []
        for pair in duplicate_pairs:
            index_to_leave.append(pair[0])
            index_to_leave.append(pair[1])
        res = df.take(index_to_leave)
        res = res.assign(id = index_to_leave)
        res = res.assign(is_deleted = [""]*len(res))
        for idx, old_ind in enumerate(index_to_leave):
            res["is_deleted"].values[idx] = "left" if idx%2 != index_to_delete else "deleted"
        return res

    def get_df_for_duplicates_two_dfs(self, df1,df2, duplicate_pairs, index_to_delete):
        df = pd.concat([df1, df2], sort=False)
        return self.get_df_for_duplicates(df, duplicate_pairs, index_to_delete)

    def get_ids_to_delete(self, duplicate_pairs, index):
        duplic_set = set()
        for pair in duplicate_pairs:
            duplic_set.add(pair[index])
        return duplic_set

    def find_duplicate_articles_by_title(self, df1, df2):
        if len(df2) == 0:
            return set()
        first_dict = self.get_articles_dict_by_title(df1)
        second_dict = self.get_articles_dict_by_title(df2, len(df1)) 
        tfidf_matrix = self.get_tf_idf_matrix(list(df1[self.title_column].values) + (list(df2[self.title_column].values) if len(df2) > 0 else []))
        tfidf_matrix_abstract = self.get_tf_idf_matrix(list(df1[self.abstract_column].values) + (list(df2[self.abstract_column].values) if len(df2) > 0 else []))
        duplicates_pairs = set()
        for year in first_dict:
            print("Year %d: %d items"%(year, len(first_dict[year])))
            for key, values in first_dict[year].items():
                if year in second_dict and key in second_dict[year]:
                    set_duplicates_first = self.get_duplicate_articles_by_title(values, second_dict[year][key], \
                                                                                                  tfidf_matrix, tfidf_matrix_abstract, df1, df2)
                    duplicates_pairs = duplicates_pairs.union(set_duplicates_first)
        return duplicates_pairs

    def remove_duplicates_two_datasets(self, df1, df2, duplicates_pairs):
        duplic_set = self.get_ids_to_delete(duplicates_pairs,0)
        self.append_duplicates_into_file(self.get_df_for_duplicates_two_dfs(df1, df2, duplicates_pairs,0), "duplicate_in_two_dataset.xlsx")
        indexes_to_keep = set(range(len(df1.index))) - duplic_set
        df1 = df1.take(list(indexes_to_keep))
        return df1

    def get_priority(self, dataset_name):
        datasets_names = {"Science journals":1,"Scopus":1, "ISI":2, "CABI":3, "CAB":4,"Web of Science":5, "Econlit":6, "Proquest":7, "Agricola":8}
        if dataset_name in datasets_names:
            return datasets_names[dataset_name]
        return 9

    def get_pair_if_duplicate(self, i,j, tfidf_matrix, tfidf_matrix_abstract, df):
        first_title = " ".join(text_normalizer.get_normalized_text_with_numbers(df[self.title_column].values[i]))
        second_title = " ".join(text_normalizer.get_normalized_text_with_numbers(df[self.title_column].values[j]))
        if (utils.normalized_levenshtein_score(
                first_title, second_title) >= 0.9 and (cosine_similarity(tfidf_matrix_abstract[i], tfidf_matrix_abstract[j]) >= 0.85 or df[self.abstract_column].values[i].strip() == "" or df[self.abstract_column].values[j].strip() == "")) or\
                         (cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]) >= 0.95\
                        and (cosine_similarity(tfidf_matrix_abstract[i], tfidf_matrix_abstract[j]) >= 0.85 or df[self.abstract_column].values[i].strip() == "" or df[self.abstract_column].values[j].strip() == "")) or\
                        (cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]) > 0.8 and \
                cosine_similarity(tfidf_matrix_abstract[i], tfidf_matrix_abstract[j]) >= 0.94):
                        return (j,i) if self.dataset_column in df.columns and self.get_priority(df[self.dataset_column].values[i]) <= self.get_priority(df[self.dataset_column].values[j]) else (i,j)
        return None

    def remove_duplicates_in_one_df_by_title(self, df, column_names_to_merge):
        year_dict = self.get_articles_dict_by_title(df)
        tfidf_matrix = self.get_tf_idf_matrix(list(df[self.title_column].values))
        tfidf_matrix_abstract = self.get_tf_idf_matrix(list(df[self.abstract_column].values))
        duplicates_all = set()
        for year in year_dict:
            id_check = 0
            print("Year %d: %d items"%(year, len(year_dict[year])))
            for key, values in year_dict[year].items():
                id_check += 1
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        pair_duplicates = self.get_pair_if_duplicate(values[i], values[j], tfidf_matrix, tfidf_matrix_abstract,df)
                        if pair_duplicates != None:
                            duplicates_all.add(pair_duplicates)
        print("Checked all articles")
        df = self.merge_df_by_columns(df, df, duplicates_all, column_names_to_merge)

        duplic_set = self.get_ids_to_delete(duplicates_all,0)
        self.append_duplicates_into_file(self.get_df_for_duplicates(df, duplicates_all,0), "duplicate_in_one_dataset.xlsx")
        print("Merged file")
        indexes_to_keep = set(range(len(df.index))) - duplic_set
        df = df.take(list(indexes_to_keep))
        print("Filtered dataset")
        return df

    def get_articles_dict_by_title(self, df, id_shift = 0):
        first_word_articles = {}
        for i in range(len(df)):
            year = df[self.year_column].values[i]  
            if year not in first_word_articles:
                first_word_articles[year] = {}
            words = text_normalizer.get_normalized_text_with_numbers(df[self.title_column].values[i])
            if len(words) == 0:
                continue
            key_word = words[0] if len(words) == 1 else words[0] + " " + words[1]
            if key_word not in first_word_articles[year]:
                first_word_articles[year][key_word] = []
            first_word_articles[year][key_word].append(i+id_shift)
        return first_word_articles

    def merge_df_by_columns(self, new_dataset, articles_df, duplicate_pairs, column_names_to_merge):
        for column_name in column_names_to_merge:
            if column_name not in articles_df.columns:
                articles_df[column_name] = ""
                for i in range(len(articles_df)):
                    articles_df[column_name].values[i] = []
            else:
                for i in range(len(articles_df)):
                    if type(articles_df[column_name].values[i]) != list:
                        articles_df[column_name].values[i] = [articles_df[column_name].values[i]]
                for i in range(len(new_dataset)):
                    if type(new_dataset[column_name].values[i]) != list:
                        new_dataset[column_name].values[i] = [new_dataset[column_name].values[i]]
            for doc in duplicate_pairs:
                articles_df[column_name].values[doc[1]-len(new_dataset)].extend(new_dataset[column_name].values[doc[0]])
                articles_df[column_name].values[doc[1]-len(new_dataset)] = list(set(articles_df[column_name].values[doc[1]-len(new_dataset)]))
        return articles_df

    def deduplicate_and_process_dataset_with_merge(self, new_dataset, articles_df, column_names_to_merge = ["team_tags"]):
        start_time = time()
        new_dataset = self.remove_duplicates_in_one_df_by_title(new_dataset, column_names_to_merge)
        print(time()-start_time)
        print("deduplication within the dataset by title is done: %d articles" % len(new_dataset))

        start_time = time()
        duplicate_pairs = self.find_duplicate_articles_by_title(new_dataset,articles_df)
        articles_df = self.merge_df_by_columns(new_dataset, articles_df, duplicate_pairs, column_names_to_merge)
        new_dataset = self.remove_duplicates_two_datasets(new_dataset,articles_df,duplicate_pairs)
        print(time()-start_time)
        print("After deduplication: %d articles"%len(new_dataset))
        
        articles_df = pd.concat([articles_df, new_dataset], sort=False)
        print("Concatenated new dataset: %d articles"%len(articles_df))
        return articles_df