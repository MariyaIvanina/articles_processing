import sys
sys.path.append('../src')

from text_processing import text_normalizer
from text_processing import search_engine_insensitive_to_spelling
from utilities import excel_writer
from utilities import excel_reader
from interventions_labeling_lib import interventions_dictionary_deriver
from text_processing import abbreviations_resolver
from interventions_labeling_lib import intervention_labels
from interventions_labeling_lib import intervention_labeling
from interventions_labeling_lib.intervention_labels import InterventionLabels
import shutil

import argparse
import pandas as pd
import pickle
import os
from datetime import datetime

def label_interventions(folder_dataset, word2vec_model_folder, interventions_model_folder, output_predicted_interv_folder):
    excelWriter = excel_writer.ExcelWriter()
    intervention_labeler = intervention_labeling.InterventionLabeling(google_models_folder=word2vec_model_folder)
    print("Initialized")
    intervention_labeler.load_previous_models(interventions_model_folder)
    print("Model loaded")
    interventions_df_full = pd.DataFrame()
    for filename in os.listdir(folder_dataset):
        if os.path.isdir(os.path.join(folder_dataset, filename)):
            continue
        interventions_df = pd.read_excel(os.path.join(folder_dataset, filename)).fillna("")
        print(filename, len(interventions_df))
        interventions_df["Predicted Label"] = [
            InterventionLabels.INTERVENTION_NUMBER_TO_LABEL[label] for label in intervention_labeler.predict_class(interventions_df.values)]
        interventions_df_full = pd.concat([interventions_df_full, interventions_df], axis=0)
        if output_predicted_interv_folder:
            excelWriter.save_df_in_excel(interventions_df, os.path.join(output_predicted_interv_folder, filename),
                column_interventions=['Label',"Predicted Label"], column_probabilities=['narrow concept mentioned in articles',
               'broad concept and narrow concept occured together in articles',
               'common topic frequency', 'our dataset word weight'],column_outliers=[("Words count", 1),("Is abbreviation", 1)])
    return interventions_df_full

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dataset', default = "../tmp/interventions_big_dataset/labeled", help='folder with interventions')
    parser.add_argument('--output_file', default = "../tmp/interventions_new.xlsx",help='file for saving')
    parser.add_argument('--search_index_folder', default = "../model/search_index", help="folder with search inverted index")
    parser.add_argument('--interventions_model_folder', default = "../model/intervention_labels_model", help="folder with interventions model")
    parser.add_argument('--word2vec_model_folder', default = "../model/synonyms_retrained_new", help="folder with word2vec embeddings")
    parser.add_argument('--abbreviations_folder', default = "../model/abbreviations_dicts", help="folder with abbreaviations")
    parser.add_argument('--output_predicted_interv_folder', default='')
    parser.add_argument('--derive_taxonomy', default='True')

    args = parser.parse_args()

    print("Interventions files folder: %s"%args.folder_dataset)
    print("File to save: %s"%args.output_file)
    print("Folder with search: %s"%args.search_index_folder)
    print("Interventions model folder: %s"%args.interventions_model_folder)
    print("Word2vec model folder: %s"%args.word2vec_model_folder)
    print("Abbreviaitons folder: %s"%args.abbreviations_folder)
    print("Output files into folder: %s"%args.output_predicted_interv_folder)
    print("Derive taxonomy: %s"%args.derive_taxonomy)

    if args.output_predicted_interv_folder and not os.path.exists(args.output_predicted_interv_folder):
        os.makedirs(args.output_predicted_interv_folder)

    interventions_df_full = label_interventions(
        args.folder_dataset, args.word2vec_model_folder, args.interventions_model_folder, args.output_predicted_interv_folder)

    if args.derive_taxonomy.lower() == "true":
        _abbreviations_resolver = abbreviations_resolver.AbbreviationsResolver([])
        _abbreviations_resolver.load_model(args.abbreviations_folder)

        joined_search_engine_inverted_index = search_engine_insensitive_to_spelling.SearchEngineInsensitiveToSpelling(load_abbreviations = True)
        joined_search_engine_inverted_index.load_model(args.search_index_folder)
        _dict_deriver = interventions_dictionary_deriver.InterventionsDictionaryDeriver(
            google_model_dir = os.path.join(args.word2vec_model_folder, "google_plus_our_dataset"))
        _dict_deriver.create_dictionary(joined_search_engine_inverted_index, _abbreviations_resolver, interventions_df=interventions_df_full)
        _dict_deriver.save_dictionary(args.output_file)