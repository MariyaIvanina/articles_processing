from sklearn.utils import shuffle
import os
from text_processing import text_normalizer
from utilities import excel_writer
from utilities import excel_reader
import pandas as pd
from study_design_type import study_type_labeler_window_agg
from study_design_type.study_type_labels import StudyTypeLabels

class FullLogicStudytypeLabeler:

    def __init__(self):
        self.exact_keywords = {"systematic review":"Systematic review", "meta analysis":"Meta analysis", "chapter":"Book chapter",\
                  "greenhouse study":"Greenhouse study", "glasshouse study":"Greenhouse study",\
                  "field study":"Field study","field survey":"Observational study","field site":"Field study", "field experiment":"Field study", "experiment station":"Field study",\
                  "greenhouse - greenhouse gas":"Greenhouse study", "glasshouse":"Greenhouse study",\
                  "laboratory study":"Laboratory study","laboratory experiment":"Laboratory study","lab study":"Laboratory study","lab experiment":"Laboratory study","lab condition":"Laboratory study","laboratory condition":"Laboratory study","laboratory scale":"Laboratory study","lab scale":"Laboratory study",\
                  "modeling study":"Modeling study", "simulation study":"Modeling study", "analysis study":"Modeling study",\
                  "participant study":"Observational study", "respondent study":"Observational study", "population study":"Observational study","farmer study":"Observational study","smallholder study":"Observational study",\
                  "participant survey":"Observational study", "respondent survey":"Observational study", "population survey":"Observational study","farmer survey":"Observational study","smallholder survey":"Observational study",\
                  "randomized controlled trial": "Observational study", "RCT": "Observational study", "quasi experimental design": "Observational study","propensity score":"Observational study","IV approaches":"Observational study",\
                  "endogenous switching":"Observational study", "RDD":"","impact evaluation":"Observational study", "comparison study":"Observational study", "distributional impact":"Observational study",\
                  "randomized controlled":"Observational study","ex post facto": "Observational study","quasi experimental":"Observational study", "ESRMs": "Observational study",\
                  "review paper":"Review paper"}
        self.key_words = {"systematic review":"Systematic review", "meta analysis":"Meta analysis", "chapter":"Book chapter",\
              "greenhouse study":"Greenhouse study", "glasshouse study":"Greenhouse study",\
              "field study":"Field study","field survey":"Observational study","field site":"Field study", "field experiment":"Field study", "experiment station":"Field study",\
              "greenhouse - greenhouse gas":"Greenhouse study", "glasshouse":"Greenhouse study",\
              "laboratory study":"Laboratory study","laboratory experiment":"Laboratory study","lab study":"Laboratory study","lab experiment":"Laboratory study","lab condition":"Laboratory study","laboratory condition":"Laboratory study","laboratory scale":"Laboratory study","lab scale":"Laboratory study",\
              "modeling study":"Modeling study", "simulation study":"Modeling study", "analysis study":"Modeling study",\
              "participant study":"Observational study", "respondent study":"Observational study", "population study":"Observational study","farmer study":"Observational study","smallholder study":"Observational study",\
              "participant survey":"Observational study", "respondent survey":"Observational study", "population survey":"Observational study","farmer survey":"Observational study","smallholder survey":"Observational study",\
              "participatory approach":"Observational study", "participatory extension approach" :"Observational study",\
              "randomized controlled trial": "Observational study", "RCT": "Observational study", "quasi experimental design": "Observational study","propensity score":"Observational study","IV approaches":"Observational study",\
              "endogenous switching":"Observational study", "RDD":"","impact evaluation":"Observational study", "comparison study":"Observational study", "distributional impact":"Observational study",\
              "randomized controlled":"Observational study","ex post facto": "Observational study","quasi experimental":"Observational study", "ESRMs": "Observational study",\
              "review paper":"Review paper",\
              "laboratory":"Laboratory study","lab":"Laboratory study",\
              "modeling":"Modeling study", "analysis":"Modeling study", "simulator":"Modeling study", "simulate":"Modeling study",\
              "interview":"Observational study", "interviewed":"Observational study","respondent":"Observational study","questionnaire":"Observational study",\
              "review":"Review paper", "reviewed":"Review paper"}

    def fill_study_type(self, study_type_df, search_engine_inverted_index,
    		column_first, column_second, exact_keywords, multilabel=True):
        study_type_df[column_first] = ""
        study_type_df[column_second] = ""
        for i in range(len(study_type_df)):
            study_type_df[column_first].values[i] = set()
            study_type_df[column_second].values[i] = set()
        for key_word in exact_keywords:
            if "-" in key_word:
                new_key_words = key_word.split("-")
                main_key_word = new_key_words[0].strip()
                articles = set(search_engine_inverted_index.find_articles_with_keywords([main_key_word], extend_with_abbreviations=False))
                for key in new_key_words[1:]:
                    articles = articles - set(search_engine_inverted_index.find_articles_with_keywords([key.strip()], extend_with_abbreviations=False))
            else:
                articles = search_engine_inverted_index.find_articles_with_keywords([key_word], extend_with_abbreviations=False)
            for art_id in articles:
                study_type_df[column_first].values[art_id].add(key_word.split("-")[0].strip())
                if multilabel or len(study_type_df[column_second].values[art_id]) == 0:
                    study_type_df[column_second].values[art_id].add(exact_keywords[key_word])
        for i in range(len(study_type_df)):
            study_type_df[column_first].values[i] = list(study_type_df[column_first].values[i])
            if multilabel:
            	study_type_df[column_second].values[i] = list(study_type_df[column_second].values[i])
            	if len(study_type_df[column_second].values[i]) == 0:
            		study_type_df[column_second].values[i] = ['No category']
            else:
	            if len(study_type_df[column_second].values[i]) == 0:
	                study_type_df[column_second].values[i] = 'No category'
        return study_type_df

    def count_category_results(self, study_type_df, column):
        cnt = 0
        dict_with_count = {}
        for i in range(len(study_type_df)):
            cnt += 1 if study_type_df[column].values[i] != "No category" else 0
            if study_type_df[column].values[i] not in dict_with_count:
                dict_with_count[study_type_df[column].values[i]] = 0
            dict_with_count[study_type_df[column].values[i]] += 1
        return cnt, dict_with_count

    def split_train_test(self, df, column, categories, split_percent = 0.05):
        train,test = pd.DataFrame(), pd.DataFrame()
        for category in categories:
            part_df = df[df[column]==category]
            part_df = shuffle(part_df)
            quantity = int((1-split_percent)*len(part_df))
            train = pd.concat([train, part_df[:quantity]],sort=False)
            test = pd.concat([test, part_df[quantity:]],sort=False)
        return train, test

    def split_train_test_left_df(self, df, column, categories, split_percent = 0.05):
        train,test = pd.DataFrame(), pd.DataFrame()
        for cnt, category in categories:
            part_df = pd.DataFrame(df[df[column]==category].values[:cnt], columns = df.columns)
            part_df = shuffle(part_df)
            quantity = int((1-split_percent)*len(part_df))
            train = pd.concat([train, part_df[:quantity]],sort=False)
            test = pd.concat([test, part_df[quantity:]],sort=False)
        return train, test

    def filter_dataset_from_test_labeled_data(self, test_dataset_filename, study_type_df):
        test_data = excel_reader.ExcelReader().read_df_from_excel(test_dataset_filename)
        test_data_dict = {}
        for i in range(len(test_data)):
            test_data_dict[test_data["Text"].values[i]] = test_data["Jaron's comments"].values[i].strip()
        print("Articles number in test dataset: %d"%len(test_data_dict))

        indices_to_remove = []
        for i in range(len(study_type_df)):
            if study_type_df["title"].values[i]+"."+study_type_df["abstract"].values[i] in test_data_dict:
                indices_to_remove.append(i)
            elif text_normalizer.remove_html_tags(study_type_df["title"].values[i]+"."+study_type_df["abstract"].values[i]) in test_data_dict:
                indices_to_remove.append(i)
        indices_to_keep = set(range(len(study_type_df.index))) - set(indices_to_remove)
        study_type_df_filtered = study_type_df.take(list(indices_to_keep))
        print("Filtered study df articles number %d"%len(study_type_df_filtered))
        print("Filtered test articles number %d"%len(indices_to_remove))
        return study_type_df_filtered

    def identify_train_test_datasets(self, study_type_df_filtered):
        train,test = self.split_train_test(study_type_df_filtered, "exact_study_type", ['Observational study','Field study','Modeling study','Laboratory study','Review paper'])
        print(train["exact_study_type"].value_counts())
        print(test["exact_study_type"].value_counts())

        train_1, test_1 = self.split_train_test_left_df(study_type_df_filtered[study_type_df_filtered["exact_study_type"]=='No category'],\
         "extended_study_type", [(1400,'Observational study'),(2300,'Modeling study'),(1600,'Laboratory study'),(2500,'Review paper')])
        print(train_1["extended_study_type"].value_counts())
        print(test_1["extended_study_type"].value_counts())

        train_full = pd.concat([train, train_1],sort=False)
        test_full = pd.concat([test, test_1],sort=False)
        print(train_full["extended_study_type"].value_counts())
        print(test_full["extended_study_type"].value_counts())

        no_category = study_type_df_filtered[study_type_df_filtered["extended_study_type"] == "No category"]
        no_category_part = no_category[no_category["abstract"] == ""]
        no_category_part_1 = pd.DataFrame(no_category_part.values[:100], columns = no_category_part.columns)
        no_category = no_category[no_category["abstract"] != ""]

        indices_to_take = []
        for i in range(len(no_category)):
            if "report" in no_category["title"].values[i].lower() or "draft" in no_category["title"].values[i].lower() or\
            "conference" in no_category["abstract"].values[i].lower() or "conference" in no_category["title"].values[i].lower() or\
            "draft" in no_category["abstract"].values[i].lower():
                indices_to_take.append(i)
        no_category_1 = no_category.take(indices_to_take[:1000])
        no_category = no_category.take(list(set(range(len(no_category.index))) - set(indices_to_take[:1000])))

        no_category = shuffle(no_category)
        no_category = pd.concat([no_category_part_1, no_category_1, no_category[:1400]],sort=False)
        no_category = shuffle(no_category)
        print("Articles number with label No category %d"%len(no_category))

        train_full = pd.concat([train_full, no_category[:2400]],sort=False)
        test_full = pd.concat([test_full, no_category[2400:]],sort=False)
        print(train_full["extended_study_type"].value_counts())
        print(test_full["extended_study_type"].value_counts())

        train_full = shuffle(train_full)
        test_full = shuffle(test_full)
        return train_full, test_full

    def find_left_dataset_without_labels(self, study_type_df_filtered, train_full):
        train_vals_dict = set()
        for i in range(len(train_full)):
            train_vals_dict.add(train_full["title"].values[i] + "." + train_full["abstract"].values[i])
        indices_to_keep = []
        for i in range(len(study_type_df_filtered)):
            text = study_type_df_filtered["title"].values[i] + "." + study_type_df_filtered["abstract"].values[i]
            if text not in train_vals_dict:
                indices_to_keep.append(i)
        left_dataset = study_type_df_filtered.take(indices_to_keep)
        left_dataset = left_dataset[left_dataset["abstract"] != ""]
        left_dataset = shuffle(left_dataset)
        print("Left dataset articles number %d"%len(left_dataset))
        return left_dataset

    def find_left_dataset_by_train_and_test_sets(self, articles_df, filename_train, filenmae_test, filename_left_dataset):
        train_df = excel_reader.ExcelReader().read_df_from_excel(filename_train)
        test_df = excel_reader.ExcelReader().read_df_from_excel(filenmae_test)
        df = pd.concat([train_df, test_df],axis=0)
        df_titles = set()
        for i in range(len(df)):
            df_titles.add(" ".join(text_normalizer.get_normalized_text_with_numbers(re.sub("\s+"," ",df["title"].values[i]).strip().lower())))
        left_ind = []
        for i in range(len(articles_df)):
            if " ".join(text_normalizer.get_normalized_text_with_numbers(re.sub("\s+"," ",articles_df["title"].values[i]).strip().lower())) not in df_titles:
                left_ind.append(i)
        left_df = articles_df.take(left_ind)
        excel_writer.ExcelWriter().save_df_in_excel(left_df, filename_left_dataset)
        return left_df

    def create_train_test_datasets(self, all_dataset_filename, test_dataset_filename, folder_with_files, search_engine_inverted_index):
        if not os.path.exists(folder_with_files):
            os.makedirs(folder_with_files)

        study_type_df = excel_reader.ExcelReader().read_df_from_excel(all_dataset_filename)

        study_type_df = self.fill_study_type(study_type_df, search_engine_inverted_index, "study_type","exact_study_type", self.exact_keywords)
        print(self.count_category_results(study_type_df, "exact_study_type"))
        study_type_df = self.fill_study_type(study_type_df, search_engine_inverted_index, "keywords_study_type","extended_study_type", self.key_words)
        print(self.count_category_results(study_type_df, "extended_study_type"))

        study_type_df_filtered = self.filter_dataset_from_test_labeled_data(test_dataset_filename, study_type_df)
        train_full, test_full = self.identify_train_test_datasets(study_type_df_filtered)
        left_dataset = self.find_left_dataset_without_labels(study_type_df_filtered, train_full)

        excel_writer.ExcelWriter().save_df_in_excel(train_full,os.path.join(folder_with_files, "study_type_all_train.xlsx"))
        excel_writer.ExcelWriter().save_df_in_excel(test_full,os.path.join(folder_with_files, "study_type_all_test.xlsx"))
        excel_writer.ExcelWriter().save_df_in_excel(left_dataset,os.path.join(folder_with_files, "study_type_all_left_dataset.xlsx"))

    def label_df_with_study_type(self, articles_df, search_engine_inverted_index, use_prediction = False, gpu_device_num = 0,
            folder="study_type_multi",
            meta_folder="study_type_multi_meta_agg",
            scibert_model_folder="../tmp/scibert_scivocab_uncased"):
        articles_df = self.fill_study_type(articles_df, search_engine_inverted_index, "keywords_study_type", "study_type", self.exact_keywords)
        study_type_labels = ['Observational study', 'Modeling study', 'Laboratory study', 'Review paper', 'Field study']
        if use_prediction:
            _study_type_labeler = study_type_labeler_window_agg.StudyTypeLabeler(
                  folder,
                  max_seq_length = 256, batch_size = 16,
                  gpu_device_num_hub = 0, gpu_device_num = 0,
                  model_folder = scibert_model_folder,
                  label_list = list(range(len(study_type_labels))),
                  meta_model_folder = meta_folder,
                  use_one_layer=True,
                  keep_prob=0.8,
                  epochs_num=4,
                  multilabel=True
                  )
                
            r = _study_type_labeler.predict_with_meta_model(articles_df[["title", "abstract"]], with_head_tail = True)
            for i in range(len(r[0])):
                for idx, label in enumerate(study_type_labels):
                    if r[0][i][idx] >= 0.5:
                        articles_df["study_type"].values[i].append(label)
            
            for i in range(len(articles_df)):
                articles_df["study_type"].values[i] = list(set(articles_df["study_type"].values[i]) - set(["No category"]))
                if not len(articles_df["study_type"].values[i]):
                    articles_df["study_type"].values[i] = ["No category"]

            
        return articles_df


