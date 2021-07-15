import sys
sys.path.append('../src')

from synonyms_module import synonyms_training
import argparse
import pandas as pd
from utilities import excel_reader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_or_folder_with_dataset', default="big_dataset_distributed", help='folder for dataset')
    parser.add_argument('--folder_to_save',default="../model/synonyms_trained_with_abbreviations_big_new", help='folder for saving')
    parser.add_argument('--all_models', dest='all_models', action='store_true')
    parser.set_defaults(all_models=False)
    parser.add_argument('--google_model', dest='google_model', action='store_true')
    parser.set_defaults(google_model=False)
    parser.add_argument('--google_model_2_3_grams', dest='google_model_2_3_grams', action='store_true')
    parser.set_defaults(google_model_2_3_grams=False)
    parser.add_argument('--fast_text', dest='fast_text', action='store_true')
    parser.set_defaults(all_models=False)
    parser.add_argument('--columns_to_use_for_sentences',default="")
    parser.add_argument('--columns_to_use_as_keywords',default="")
    

    args = parser.parse_args()
    print("File or Folder with dataset: %s"%args.file_or_folder_with_dataset)
    print("Folder to save: %s"%args.folder_to_save)
    print("All models: %s"%args.all_models)
    print("Only google model: %s"%args.google_model)
    print("Only google_model_2_3_grams: %s"%args.google_model_2_3_grams)
    print("Only Fast text model: %s"%args.fast_text)
    print("Columns to use for sentences: %s"%args.columns_to_use_for_sentences)
    print("Columns to use as keywords: %s"%args.columns_to_use_as_keywords)

    args.columns_to_use_for_sentences = [col.strip() for col in args.columns_to_use_for_sentences.split(",") if col.strip()]
    args.columns_to_use_as_keywords = [col.strip() for col in args.columns_to_use_as_keywords.split(",") if col.strip()]
    all_columns = list(set(args.columns_to_use_for_sentences + args.columns_to_use_as_keywords))

    articles_df = excel_reader.ExcelReader().read_df(args.file_or_folder_with_dataset)
    articles_df = articles_df[all_columns]
    print("Dataset is loaded ...")

    synonyms_trainer = synonyms_training.SynonymsTrainer(args.folder_to_save,
        columns_to_use_for_sentences=args.columns_to_use_for_sentences,
        columns_to_use_as_keywords=args.columns_to_use_as_keywords)
    synonyms_trainer.train_phrases(articles_df)
    synonyms_trainer.prepare_sentences_for_learning(articles_df)
    synonyms_trainer.train_models(all_models=args.all_models, google_model=args.google_model,
        google_model_2_3_grams=args.google_model_2_3_grams, fast_text=args.fast_text)
    synonyms_trainer.save_phrases_inverted_index_and_popular_expressions(args.folder_to_save)