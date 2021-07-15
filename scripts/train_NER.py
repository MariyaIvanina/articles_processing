import pandas as pd
import requests
import json
import re
import sys
import numpy as np
import os
import nltk
import spacy
import pickle

sys.path.append('../src')
from commons import elastic

from utilities import excel_reader
import argparse
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
import random
from tqdm import tqdm # loading bar

from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(model, examples):
    scorer = Scorer()
    cnt = 0
    for input_, annot in examples:
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = model(input_)
        if len(annot['entities']) == 0 and len(
                [(ent.text, ent.label_) for ent in pred_value.ents]) == 0:
            scorer.ner.tp += 1
            for label in scorer.ner_per_ents:
                scorer.ner_per_ents[label].tp += 1
            cnt += 1
        else:
            try:
                scorer.score(pred_value, gold)
            except:
                print("Error while predicting: ")
                print(input_)
                pass
    return scorer.scores

def split_outcomes_data(merged_data):
    train_data = []
    train_data_outcomes = []
    train_sentences = []
    for data in merged_data:
        train_data.append((data[0], data[1]))
        train_data_outcomes.extend(data[2])
        train_sentences.append((data[0], [d[1] for d in data[2]]))
    return train_data, train_data_outcomes, train_sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_with_dataset')
    parser.add_argument('--folder_to_save')
    parser.add_argument('--n_iter', default="100")
    
    args = parser.parse_args()
    print("Folder with dataset: %s"%args.folder_with_dataset)
    print("Folder to save: %s"%args.folder_to_save)
    print("Number of itertions: %s"%args.n_iter)
    args.n_iter = int(args.n_iter)

    basename = os.path.basename(args.folder_to_save)
    train_data_file = os.path.join(
        os.path.dirname(
            args.folder_to_save), basename + "_train_data")
    metrics_data_file = os.path.join(
        os.path.dirname(
            args.folder_to_save), basename + "_metrics")

    if os.path.exists(train_data_file):
        TRAIN_DATA, outcomes_train_label_data, train_sentences, EVAL_DATA, outcomes_eval_label_data, eval_sentences = pickle.load(open(train_data_file, "rb"))
    else:
        classes_to_take = set()
        for file in os.listdir(args.folder_with_dataset):
            df_outcome_mary = excel_reader.ExcelReader().read_df_from_excel(
                    os.path.join(args.folder_with_dataset, file))
            columns = set(list(df_outcome_mary.columns)) - set(["Sentence"])
            classes_to_take.update(columns)
        classes_to_take = list(classes_to_take)
        print("Classes to take: ", classes_to_take)


        all_data = []
        y = []
        for file in os.listdir(args.folder_with_dataset):
            df_outcome_mary = excel_reader.ExcelReader().read_df_from_excel(
                    os.path.join(args.folder_with_dataset, file))
            for i in range(len(df_outcome_mary)):
                outcomes = []
                outcomes_per_label = []
                outcomes_labels = []
                for column in classes_to_take:
                    outcomes_found = set([outcome for outcome in df_outcome_mary[column].values[i].split(";") if outcome.strip()])
                    outcomes.extend(outcomes_found)
                    for outcome in outcomes_found:
                        outcomes_per_label.append((outcome, column))
                    
                    outcomes_labels.append(1 if len(outcomes_found) else 0)
                y.append(outcomes_labels)
                outcomes_found = set()
                for sent in [df_outcome_mary["Sentence"].values[i]]:
                    found_entities = []
                    found_outcome_plus_labels = []
                    for outcome, label in outcomes_per_label:
                        outcome = outcome.strip().strip(",").strip(".").strip()
                        if outcome.lower() in sent.lower():
                            start_ind = sent.lower().rfind(outcome.lower())
                            end_ind = start_ind + len(outcome)
                            if sent[start_ind: end_ind].lower() != outcome.lower():
                                print(sent)
                                print(start_ind, end_ind)
                                print(sent[start_ind: end_ind])
                                print(sent[start_ind: end_ind].lower(), outcome.lower())
                            found_entities.append((start_ind, end_ind, 'OUTCOME'))
                            outcomes_found.add(outcome)
                            found_outcome_plus_labels.append((outcome, label))
                    all_data.append((sent, {'entities': found_entities}, found_outcome_plus_labels))
                    
                if len(outcomes_found) != len(set(outcomes)):
                    print(outcomes, outcomes_found)
                    print(i)
        print("All data ", len(all_data))

        np.random.seed(1237)
        np.random.shuffle(all_data)
        np.random.seed(1237)
        np.random.shuffle(y)

        X_train, y_train, X_test, y_test = iterative_train_test_split(
            np.asarray(all_data), np.asarray(y), test_size = 0.1)
                

        TRAIN_DATA, outcomes_train_label_data, train_sentences = split_outcomes_data(X_train)
        EVAL_DATA, outcomes_eval_label_data, eval_sentences = split_outcomes_data(X_test)
        
        pickle.dump([TRAIN_DATA, outcomes_train_label_data, train_sentences,
            EVAL_DATA, outcomes_eval_label_data, eval_sentences], open(train_data_file, "wb"))

    print("Train data", len(TRAIN_DATA))
    print("Test data", len(EVAL_DATA))

    if not os.path.exists(args.folder_to_save):

        ner_model = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in ner_model.pipe_names:
            ner = ner_model.create_pipe('ner')
            ner_model.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner_model = ner_model.get_pipe('ner')

        for _, annotations in TRAIN_DATA:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in ner_model.pipe_names if pipe != 'ner']
        train_losses = []
        val_metrics = []
        max_val_metric = 0.0
        with ner_model.disable_pipes(*other_pipes):  # only train NER
            optimizer = ner_model.begin_training()
            for itn in range(args.n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in tqdm(TRAIN_DATA):
                    ner_model.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout 
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(itn, losses)
                train_losses.append(losses["ner"])
                val_metric = evaluate(ner_model, EVAL_DATA)
                print("Val metric ", val_metric)
                val_metrics.append(val_metric["ents_f"])
                if val_metric["ents_f"] >= max_val_metric:
                    max_val_metric = val_metric["ents_f"]
                    ner_model.to_disk(args.folder_to_save + "-%d_iter"%itn)
        ner_model.to_disk(args.folder_to_save)
        pickle.dump([train_losses, val_metrics], open(metrics_data_file, "wb"))
    else:
        ner_model = spacy.load(args.folder_to_save)  # load existing spaCy model
        print("Loaded model '%s'" % args.folder_to_save)

    print("Evaluation")
    print(evaluate(ner_model, EVAL_DATA))