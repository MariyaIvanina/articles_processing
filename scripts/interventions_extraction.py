import sys
sys.path.append('../src')

from text_processing import text_normalizer
from text_processing import search_engine_insensitive_to_spelling
from utilities import excel_writer
from utilities import excel_reader
from interventions_labeling_lib import hyponym_search
from interventions_labeling_lib import hearst_pattern_finder
from interventions_labeling_lib import hyponym_statistics
from interventions_labeling_lib import coreferenced_concepts_finder
from interventions_labeling_lib import storage_interventions_finder
from text_processing import abbreviations_resolver
import shutil

import argparse
import pandas as pd
import pickle
import os
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dataset', default = "new_big_dataset", help='folder for dataset')
    parser.add_argument('--folder', default = "interventions_found",help='folder for saving')
    parser.add_argument('--search_index_folder', default = "../model/search_index_3", help="folder with search inverted index")
    parser.add_argument('--continue_extract', dest='continue_extract', action='store_true')
    parser.set_defaults(continue_extract=False)
    parser.add_argument('--global_interv', dest='global_interv', action='store_true')
    parser.set_defaults(global_interv=False)
    parser.add_argument('--extract_hearst', dest='extract_hearst', action='store_true')
    parser.set_defaults(extract_hearst=False)
    parser.add_argument('--extract_coreference', dest='extract_coreference', action='store_true')
    parser.set_defaults(extract_coreference=False)
    parser.add_argument('--extract_SRL', dest='extract_SRL', action='store_true')
    parser.set_defaults(extract_SRL=False)
    parser.add_argument('--extract_all_methods', dest='extract_all_methods', action='store_true')
    parser.set_defaults(extract_all_methods=False)
    parser.add_argument('--use_components_keywords', dest='use_components_keywords', action='store_true')
    parser.set_defaults(use_components_keywords=False)
    parser.add_argument('--keywords_to_add', default="")
    parser.add_argument('--columns_to_use', default="title,abstract")

    args = parser.parse_args()

    print("Dataset folder: %s"%args.folder_dataset)
    print("Folder to save: %s"%args.folder)
    print("File with additional keywords: %s"%args.keywords_to_add)
    print("Columns to use: %s"%args.columns_to_use)
    print("Folder with search: %s"%args.search_index_folder)
    print("Continue extract: %s"%args.continue_extract)
    print("Extract all methods: %s"%args.extract_all_methods)
    print("Use components keywords: %s"%args.use_components_keywords)
    if not args.extract_all_methods:
        print("Extract with hearst patterns: %s"%args.extract_hearst)
        print("Extract with coreference model: %s"%args.extract_coreference)
        print("Extract with SRL model %s"%args.extract_SRL)

    args.columns_to_use = [col.strip() for col in args.columns_to_use.split(",")]

    keywords_to_add = []
    if args.keywords_to_add:
        with open(args.keywords_to_add, "r") as f:
            for line in f.readlines():
                if line.strip():
                    keywords_to_add.append(text_normalizer.normalize_sentence(line.strip()))
    print("Keywords to add: ", keywords_to_add)

    filter_word_list = text_normalizer.build_filter_dictionary(["../data/Filter_Geo_Names.xlsx"])

    _abbreviation_resolver = abbreviations_resolver.AbbreviationsResolver(filter_word_list)
    _abbreviation_resolver.load_model("../model/abbreviations_dicts")

    articles_df = excel_reader.ExcelReader().read_df(args.folder_dataset)
    articles_df_1 = articles_df[args.columns_to_use + ["id"]] if "id" in articles_df.columns else articles_df[args.columns_to_use]
    articles_df = articles_df_1
    articles_df_1 = []
    articles_df["keywords"] = ""
    articles_df["identificators"] = ""
    print("Dataset is loaded ...")

    search_engine_inverted_index = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling(load_abbreviations = True,
        columns_to_process=args.columns_to_use)
    if args.search_index_folder.strip():
        search_engine_inverted_index.load_model(args.search_index_folder)
    else:
        search_engine_inverted_index.create_inverted_index(articles_df)

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    keywords_components = ["component", "intervention", "project goal",
        "development objective", "overall goal", "project objective",
        "project rationale", "project main goal", "subcomponent", "sub component",
        "project development objective", "program purpose", "project purpose",
        "program rationale", "program main goal", "program objective" ] if args.use_components_keywords else []

    
    key_word_mappings = keywords_to_add + keywords_components + ["practice", "approach", "intervention", "input",
    "strategy", "policy", "program", "programme", "initiative", "technology",
     "science", "technique", "innovation", "biotechnology","machine", "mechanism", 
     "equipment","tractor","device","machinery","project","gadget"]
    filter_hypernyms =  keywords_components + ["practice", "approach", "intervention", 
        "input","strategy", "policy", "program", "programme", "initiative", "technology", \
        "science", "technique", "innovation", "biotechnology","machine", "mechanism", 
        "equipment","device","project"]

    folder_to_save = "../model/"
    if not args.global_interv:
        folder_to_save = os.path.join(args.folder, "model")
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    if args.extract_all_methods or args.extract_hearst:
        print("Started hyponyms extraction...")

        hyponyms_search = hyponym_search.HyponymsSearch()
        if args.continue_extract or not os.path.exists(os.path.join(folder_to_save, "hyponyms_found_big_dataset_full.pickle")):
            hyponyms_search.find_hyponyms_and_hypernyms(
                articles_df, search_engine_inverted_index, os.path.join(folder_to_save, "hyponyms_found_big_dataset_full.pickle"), columns_to_use=args.columns_to_use)
            pickle.dump(hyponyms_search, open(os.path.join(folder_to_save, "hyponyms_found_big_dataset_full.pickle"),"wb"))
        else:
            hyponyms_search = pickle.load(open(os.path.join(folder_to_save, "hyponyms_found_big_dataset_full.pickle"),"rb"))

        print("Hyponyms are extracted...")

        hs_pruned = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index, _abbreviation_resolver, hyponyms_search.dict_hyponyms,hyponyms_search,\
         filter_word_list=filter_word_list, filter_hypernyms = filter_hypernyms)
        hs_pruned.save_pruned_hyponym_pairs_to_file(os.path.join(args.folder, "hyponyms_found_cleaned.xlsx"),args.folder)

        if os.path.exists(os.path.join(folder_to_save, "hyponyms_found_1.pickle")):

            hyponyms_search = pickle.load(open(os.path.join(folder_to_save, "hyponyms_found_1.pickle"),"rb"))
            hs_pruned = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index, _abbreviation_resolver, hyponyms_search.dict_hyponyms,hyponyms_search, \
                filter_word_list=filter_word_list, filter_hypernyms = filter_hypernyms)
            hs_pruned.save_pruned_hyponym_pairs_to_file(os.path.join(args.folder, "hyponyms_found_cleaned_small.xlsx"),args.folder)

        print("Hyponyms files are saved")

    if args.extract_all_methods or args.extract_coreference:
        print("Started coreferenced hyponyms extraction ...")

        _coreferenced_concepts_finder = coreferenced_concepts_finder.CoreferencedConceptsFinder(key_word_mappings)
        _coreferenced_concepts_finder.find_coreferenced_pairs(articles_df, search_engine_inverted_index, continue_extract = args.continue_extract,\
         folder = os.path.join(folder_to_save,"coreferences_found"), file_with_gl_hyp = os.path.join(folder_to_save, "coref_hyponyms_gl.pickle"),
         column_to_use=args.columns_to_use[0])

        hs_stat = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index,_abbreviation_resolver,\
         _coreferenced_concepts_finder.hyponyms_search_part.dict_hyponyms, _coreferenced_concepts_finder.hyponyms_search_part,
            filter_word_list=filter_word_list, filter_hypernyms = filter_hypernyms)
        hs_stat.save_pruned_hyponym_pairs_to_file(os.path.join(args.folder,"coreference_interventions_cleaned_big_dataset.xlsx"), args.folder)

        if os.path.exists(os.path.join(folder_to_save, "hyponyms_found_search_1.pickle")):
            hyponyms_search_part = pickle.load(open(os.path.join(folder_to_save, "hyponyms_found_search_1.pickle"),"rb"))
            hs_stat = hyponym_statistics.HyponymStatistics(key_word_mappings, search_engine_inverted_index, _abbreviation_resolver, hyponyms_search_part.dict_hyponyms, hyponyms_search_part, filter_word_list=filter_word_list, filter_hypernyms = filter_hypernyms)
            hs_stat.save_pruned_hyponym_pairs_to_file(os.path.join(args.folder,"coreference_interventions_small.xlsx"), args.folder)

        print("Coreferenced hyponyms are saved ...")

    if args.extract_all_methods or args.extract_SRL:

        print("Started storage hyponyms extraction ...")

        words_for_finding_preventions = set(["prevent","improve","enhance","augment","maximize","strengthen","promote","boost",\
                                        "optimize","maximise","minimize","minimise","mitigate", "reduce", "avoid","stop","counteract"])

        _storage_interventions_finder = storage_interventions_finder.StorageInterventionsFinder(search_engine_inverted_index, _abbreviation_resolver,\
                                                                words_preventions = words_for_finding_preventions,\
                                                                filter_word_list = filter_word_list,\
                                                                filter_hypernyms = filter_hypernyms, folder = "../notebooks/Parsed_sentences" if args.global_interv else os.path.join(folder_to_save, "Parsed_sentences"))
        _storage_interventions_finder.load_parsed_sentences()
        _storage_interventions_finder.parse_sentences(articles_df)
        _storage_interventions_finder.save_found_interventions(os.path.join(args.folder,"storage_hyponyms_big_dataset_cleaned.xlsx"), search_engine_inverted_index,\
         key_word_mappings, args.folder, save_debug_info=True)

        print("Storage hyponyms are saved ...")

    if os.path.exists("../data/hyponyms_for_train.xlsx"):
        shutil.copy("../data/hyponyms_for_train.xlsx", args.folder)

    if os.path.exists("../data/storage_hyponyms.xlsx"):
        shutil.copy("../data/storage_hyponyms.xlsx", args.folder)