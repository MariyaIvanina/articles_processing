import pandas as pd
import pickle
import os
from text_processing import text_normalizer
from text_processing import search_engine_insensitive_to_spelling
from text_processing import author_and_affiliations_processing
from text_processing import geo_names_finder
from text_processing import population_tags_finder
from text_processing import crops_finder
from text_processing import topic_modeling
from text_processing import column_filler
from text_processing import advanced_text_normalization
from text_processing import keywords_normalizer
from text_processing import journal_normalizer
from text_processing import duplicate_finder
from text_processing import column_data_renamer
from text_processing import context_grisp_classifier
from text_processing import label_interventions
from text_processing import strategy_focus_labeller
from outcomes_modelling import outcomes_multi_label_predictor
from interventions_labeling_lib import programs_extractor
from interventions_labeling_lib import interventions_search_for_labeling
from interventions_labeling_lib import measurements_labeler
from interventions_labeling_lib import compared_terms_finder
from text_processing import text_processing_logger
from study_design_type import full_logic_study_type_labeler
from bert_models import priority_labeler
import json
from time import time
import sys, traceback

class AllColumnFiller():

    def __init__(self, log_status_filename = ""):
        self.column_classes = {
            "AdvancedTextNormalizer": self.normalize_text_fields,
            "Deduplicator": self.deduplicate_articles,
            "KeywordNormalizer": self.normalize_keywords,
            "JournalNormalizer":self.normalize_journals,
            "AuthorAndAuthorAffiliationExtractor": self.fill_author_and_author_affiliations,
            "GeoNameFinder": self.fill_geo_names,
            "TopicModeler": self.fill_topic_keywords,
            "CropsSearch": self.fill_crops,
            "PopulationTagsFinder": self.fill_population_tags,
            "ProgramExtractor": self.fill_programs,
            "ColumnFiller": self.fill_column_with_dictionary,
            "InterventionsSearchForLabeling": self.fill_interventions,
            "MeasurementsLabeler":self.fill_measurements,
            "StudyTypeLabeller":self.fill_study_type,
            "ComparedTermsLabeller":self.fill_compared_terms,
            "FullTextPriorityLabeller":self.fill_full_text_probability,
            "ColumnDataRenamer": self.fill_column_renamed_data,
            "GRIPSClassifier": self.fill_grips_classifications,
            "InterventionLabeller": self.fill_intervention_classifications,
            "OutcomesFinder": self.fill_outcomes,
            "StrategyFocusLabeller": self.fill_strategy_focus
        }
        self.log_status_filename = log_status_filename

    def deduplicate_articles(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _deduplicator = duplicate_finder.DuplicateFinder("deduplicate_%d"%time())
        return _deduplicator.remove_duplicates_in_one_df_by_title(articles_df, [] if "columns_to_merge" not in column_info else column_info["columns_to_merge"])

    def normalize_journals(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _journal_normalizer = journal_normalizer.JournalNormalizer()
        return _journal_normalizer.correct_journal_names(articles_df)

    def normalize_keywords(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        key_words_normalizer = keywords_normalizer.KeywordsNormalizer()
        return key_words_normalizer.normalize_key_words(articles_df)

    def normalize_text_fields(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        advanced_text_normalizer = advanced_text_normalization.AdvancedTextNormalizer(_abbreviations_resolver, full_normalization = column_info["full_normalization"] if "full_normalization" in column_info else False)
        return advanced_text_normalizer.normalize_text_for_df(articles_df)

    def fill_author_and_author_affiliations(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        author_and_affiliation_extractor = author_and_affiliations_processing.AuthorAndAffiliationsProcessing()
        articles_df = author_and_affiliation_extractor.process_authors_and_affiliations(articles_df, author_mapping = column_info['author_mapping'] if "author_mapping" in column_info else {"AU":"author"},\
         affiliations_mapping_info = column_info['affiliations_mapping'] if 'affiliations_mapping' in column_info else {"raw_affiliation":"affiliation"})
        return articles_df

    def fill_geo_names(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        countries_finder = geo_names_finder.GeoNameFinder()
        articles_df = countries_finder.label_articles_with_geo_names(articles_df, search_engine_inverted_index,\
         only_countries_columns = [] if "only_countries_columns" not in column_info else column_info["only_countries_columns"],
         columns_with_country_code = [] if "columns_with_country_code" not in column_info else column_info["columns_with_country_code"],
         use_cache= False if "use_cache" not in column_info else column_info["use_cache"])
        return articles_df

    def fill_topic_keywords(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        topic_modeler = topic_modeling.TopicModeler([], 0,  use_3_grams=True, train = False,
            folder_for_wordvec=("../model" if "folder_for_wordvec" not in column_info else column_info["folder_for_wordvec"]))
        topic_modeler.load_model("../model/nmf_3_grams_new_1" if "model_folder" not in column_info else column_info["model_folder"])
        topic_modeler.calculate_statistics_by_topics(search_engine_inverted_index, articles_df)
        articles_df = topic_modeler.fill_topic_for_articles(articles_df, search_engine_inverted_index, column_name="topics", topic_mode="raw topic")
        articles_df = topic_modeler.fill_topic_for_articles(articles_df, search_engine_inverted_index, column_name="topics_keywords", topic_mode="topic keywords")
        articles_df = topic_modeler.fill_topic_for_articles(articles_df, search_engine_inverted_index, column_name="topics_hierarchy", topic_mode="topic hierarchy")
        return articles_df

    def fill_crops(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        threshold = 0.92 if "threshold" not in column_info else column_info["threshold"]
        crops_search = crops_finder.CropsSearch(search_engine_inverted_index, filename = column_info["file_dictionary"], threshold = threshold, status_logger = status_logger)
        articles_df = crops_search.find_crops(
            articles_df, column_info["column_name"], keep_hierarchy=(column_info["keep_hierarchy"] if "keep_hierarchy" in column_info else False))
        return articles_df

    def fill_population_tags(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _population_tags_finder = population_tags_finder.PopulationTagsFinder(
            columns_to_process=["title","abstract","keywords","identificators"] if "columns_to_process" not in column_info else column_info["columns_to_process"])
        articles_df = _population_tags_finder.label_with_population_tags(articles_df, search_engine_inverted_index)
        return articles_df

    def fill_programs(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _programs_extractor = programs_extractor.ProgramExtractor([])
        articles_df = _programs_extractor.label_articles_with_programs(articles_df, search_engine_inverted_index, _abbreviations_resolver,
            program_filename="../data/extracted_programs.xlsx" if "program_filename" not in column_info else column_info["program_filename"],
            model_type="model" if "model_type" not in column_info else column_info["model_type"],
            model_folder="../tmp/programs_extraction_model_2619" if "model_folder" not in column_info else column_info["model_folder"],
            column_name="programs_found" if "column_name" not in column_info else column_info["column_name"],
            columns_to_process=["title", "abstract"] if "columns_to_process" not in column_info else column_info["columns_to_process"])
        return articles_df

    def fill_column_with_dictionary(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _colum_filler = column_filler.ColumnFiller(column_info["column_dictionary"], status_logger = status_logger,
            keyword_column=(column_info["keyword_column"] if "keyword_column" in column_info else "Keyword"),
            high_level_label_column=(column_info["high_level_label_column"] if "high_level_label_column" in column_info else "High level label"))
        articles_df = _colum_filler.label_articles_with_outcomes(column_info["column_name"], articles_df, search_engine_inverted_index,\
         _abbreviations_resolver, label_details_column = (column_info["column_details"] if "column_details" in column_info else ""),
         keep_hierarchy=(column_info["keep_hierarchy"] if "keep_hierarchy" in column_info else False),
         resolve_abbreviations=(column_info["resolve_abbreviations"] if "resolve_abbreviations" in column_info else False))
        return articles_df

    def fill_column_renamed_data(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _column_data_renamer = column_data_renamer.ColumnDataRenamer(values_dict=({} if "values_dict" not in column_info else column_info["values_dict"]),
            dict_filename=("" if "dict_filename" not in column_info else column_info["dict_filename"]), status_logger = status_logger)
        articles_df = _column_data_renamer.label_articles_with_renamed_values(
            ("" if "column_to_rename" not in column_info else column_info["column_to_rename"]),
            articles_df, column_to_save=(column_info["column_to_save"] if "column_to_save" in column_info else ""))
        return articles_df

    def fill_interventions(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _interventions_search_for_labeling = interventions_search_for_labeling.InterventionsSearchForLabeling(column_info["file"])
        articles_df = _interventions_search_for_labeling.find_interventions_from_dictionary(articles_df, search_engine_inverted_index, _abbreviations_resolver)
        return articles_df

    def fill_measurements(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _measurements_labeler = measurements_labeler.MeasurementsLabeler(
            "../bert/bert_results_6" if "model_folder" not in column_info else column_info["model_folder"],
            gpu_device_num = column_info["gpu_device_num"] if "gpu_device_num" in column_info else 0)
        articles_df = _measurements_labeler.label_measurement_columns(articles_df, column_info["folder_with_measurements"],\
            num_words_limit = None if "num_words_limit" not in column_info else column_info["num_words_limit"],
            num_words_limit_with_ids = None if "num_words_limit_with_ids" not in column_info else column_info["num_words_limit_with_ids"],
            mappings_for_measurements = {"interventions_found_raw":"measurements_for_interventions",\
                                    "plant_products_search_details":"measurements_for_crops",\
                                    "animal_products_search_details":"measurements_for_crops"} if "mappings_for_measurements" not in column_info else column_info["mappings_for_measurements"])
        return articles_df

    def fill_study_type(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _study_type_labeller = full_logic_study_type_labeler.FullLogicStudytypeLabeler()
        articles_df = _study_type_labeller.label_df_with_study_type(articles_df, search_engine_inverted_index, use_prediction = True,\
         gpu_device_num = column_info["gpu_device_num"] if "gpu_device_num" in column_info else 0,
         folder="study_type_multi" if "folder" not in column_info else column_info["folder"],
         meta_folder="study_type_multi_meta_agg" if "meta_folder" not in column_info else column_info["meta_folder"],
         scibert_model_folder="../tmp/scibert_scivocab_uncased" if "scibert_model_folder" not in column_info else column_info["scibert_model_folder"])
        return articles_df

    def fill_compared_terms(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _compared_terms_finder = compared_terms_finder.ComparedTermsFinder(search_engine_inverted_index, _abbreviations_resolver, "../tmp/parse_compare_%d"%time())
        articles_df = _compared_terms_finder.fill_compared_items(articles_df, search_engine_inverted_index)
        return articles_df

    def fill_full_text_probability(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _priority_labeller = priority_labeler.PrioirityLabeler("../bert/full_text_3", boost_model="../bert/boost_full_text_1",\
         gpu_device_num = column_info["gpu_device_num"] if "gpu_device_num" in column_info else 0, model_folder = "../model/scibert_scivocab_uncased")
        articles_df = _priority_labeller.label_df_with_probabilities_to_use_full_text(articles_df)
        return articles_df

    def fill_grips_classifications(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _context_grisp_classifier = context_grisp_classifier.GRIPSClassificationFiller(status_logger=status_logger,
            all_categories_file=("../tmp/ifad_documents/GRIPS_simple_table.xlsx" if "all_categories_file" not in column_info else column_info["all_categories_file"]),
            distilled_categories_train=("../tmp/ifad_documents/distilled_subcategory.xlsx" if "distilled_categories_train" not in column_info else column_info["distilled_categories_train"]),
            word_embeddings_folder=column_info["word_embeddings_folder"],
            column_to_label="context" if "column_to_label" not in column_info else column_info["column_to_label"],
            subcategory_column="subcategory" if "subcategory_column" not in column_info else column_info["subcategory_column"],
            category_column="category" if "category_column" not in column_info else column_info["category_column"])
        articles_df = _context_grisp_classifier.label_with_grisp_categories(articles_df)
        return articles_df

    def fill_outcomes(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _outcomes_multi_label_predictor = outcomes_multi_label_predictor.OutcomesMultiLabelPredictor(
            "../model/bert_exp_outcome_sentences_new_multilabel_15epoch_1300_mixed_0.7" if "model_folder" not in column_info else column_info["model_folder"])
        outcomes_found_column = "outcomes_found" if "column" not in column_info else column_info["column"]
        outcomes_details_column = "outcomes_details" if "column_details" not in column_info else column_info["column_details"]
        columns_to_take = ["title", "abstract"] if "columns_to_take" not in column_info else column_info["columns_to_take"]
        articles_df["text_temp"] = ""
        for column in columns_to_take:
            articles_df["text_temp"] = articles_df["text_temp"] + articles_df[column] + " . "
        found_labels, outcome_details = _outcomes_multi_label_predictor.predict_all_labels(articles_df["text_temp"].values)
        articles_df[outcomes_found_column] = found_labels
        articles_df[outcomes_details_column] = ["\n".join(detail) for detail in outcome_details]
        articles_df = articles_df.drop(["text_temp"], axis=1)
        return articles_df

    def fill_intervention_classifications(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _label_interventions = label_interventions.InterventionsLabeller(column_info["interventions_model_folder"],
            column_info["word2vec_model_folder"],
            status_logger=status_logger)
        return _label_interventions.label_df(articles_df, narrow_concept_column=column_info["narrow_concept_column"],
            broad_concepts_name=None if "broad_concepts_name" not in column_info else column_info["broad_concepts_name"],
            predicted_label_column="Predicted Label" if "predicted_label_column" not in column_info else column_info["predicted_label_column"])

    def fill_strategy_focus(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, column_info, status_logger = None):
        _strategy_focus_filler = strategy_focus_labeller.StrategyFocusLabeller(column_info["column_dictionary"], status_logger = status_logger)
        articles_df = _strategy_focus_filler.label_articles_with_strategies(
            column_info["column_name"], articles_df, search_engine_inverted_index,\
            _abbreviations_resolver, label_details_column = (column_info["column_details"] if "column_details" in column_info else ""))
        return articles_df

    def create_status_logger(self, column_info):
        if "step_name" in column_info and self.log_status_filename != "":
            return text_processing_logger.TextProcessingLogger(self.log_status_filename, step_name = column_info["step_name"])
        return None

    def fill_columns_for_df(self, articles_df, search_engine_inverted_index, _abbreviations_resolver, settings_filename = "", settings_json = {}):
        if settings_filename != "":
            column_settings = json.loads(open(settings_filename,"rb").read())["columns"]
        else:
            column_settings = settings_json["columns"]
        for column_info in column_settings:
            print("Started processing ", column_info)
            start_time = time()
            try:
                column_logger = self.create_status_logger(column_info)
                articles_df = self.column_classes[column_info["column_filler_class"]](articles_df, search_engine_inverted_index,_abbreviations_resolver,column_info, column_logger)
                if column_logger is not None:
                    column_logger.update_status_for_step("Finished")
            except Exception as e:
                print("Column info ", column_info, " was not processed properly")
                if column_logger is not None:
                    column_logger.update_status_for_step("Finished with errors", error = str(e))
                traceback.print_exc()
            print("Processed for {}s".format(time()-start_time))
        return articles_df

