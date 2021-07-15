from bert_models.base_bert_model import BaseBertModel
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
import time
from scipy.special import softmax
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

class BaseBertModelMetaModel(BaseBertModel):

    def __init__(self, output_dir, label_list, boost_model = "", gpu_device_num_hub=0,gpu_device_num = 1, batch_size = 16, max_seq_length = 256,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "", label_column = "label",
        use_concat_results=False, meta_model_folder = "",  use_one_layer = True, keep_prob=0.9,
        epochs_num=1, multilabel=False):
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num_hub=gpu_device_num_hub,
            gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,
            bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column,
            use_concat_results = use_concat_results, multilabel=multilabel)
        self.multilabel = multilabel
        self.meta_model_folder = meta_model_folder
        self.use_one_layer = use_one_layer
        self.keep_prob = keep_prob
        self.epochs_num = epochs_num

    def prepare_dataset_for_train(self, train, use_tail = False):
        res_prob, res_label, res_y, output_layers = self.predict_for_df(train, is_head = True, with_output_layer=True)
        res_prob, res_label, res_y, output_layers_agg = self.predict_for_df(train, is_head = True, with_output_layer=True, window_agg=True)
        if use_tail:
            res_prob_tail, res_label_tail, res_y_tail, output_layers_tail = self.predict_for_df(train, is_head = False, with_output_layer=True)
            concat_data = np.concatenate([ output_layers, output_layers_agg, output_layers_tail], axis=1)
            return concat_data
        return output_layers

    def prepare_datasets_for_meta_learning(self, train, test, study_df, use_tail = False):
        train_x = self.prepare_dataset_for_train(train, use_tail = use_tail)
        test_x = self.prepare_dataset_for_train(test, use_tail = use_tail)
        study_x = self.prepare_dataset_for_train(study_df, use_tail = use_tail)
        return train_x, test_x, study_x

    def create_meta_model(self):
        if self.use_one_layer:
            model = tf.keras.models.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(768*3,)),
                keras.layers.Dropout(1.0 - self.keep_prob),
                keras.layers.Dense(len(self.label_list), input_shape=(128,), activation=('sigmoid' if self.multilabel else 'relu'))
            ])
        else:
            model = tf.keras.models.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(768*3,)),
                keras.layers.Dense(32, activation='relu', input_shape=(128,)),
                keras.layers.Dropout(1.0 - self.keep_prob),
                keras.layers.Dense(len(self.label_list), input_shape=(32,), activation=('sigmoid' if self.multilabel else 'relu'))
            ])

        if self.multilabel:
            model.compile(optimizer='adam',
                loss='binary_crossentropy'
                )
        else:
            model.compile(optimizer='adam',
                        loss="sparse_categorical_crossentropy",
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        return model

    def prepare_multilabel_target(self, df, column):
        list_labels = []
        for i in range(len(df)):
            list_labels.append(df[column].values[i])
        return np.asarray(list_labels)

    def train_meta_model(self, train, test, use_tail = False, for_train = True):
        train_x = self.prepare_dataset_for_train(train, use_tail = use_tail)
        test_x = self.prepare_dataset_for_train(test, use_tail = use_tail)
        if self.multilabel:
            train_y = self.prepare_multilabel_target(train, self.label_column)
            test_y = self.prepare_multilabel_target(test, self.label_column)
        else:
            train_y = train[self.label_column].values
            test_y = test[self.label_column].values

        model = self.create_meta_model()

        model.fit(train_x, 
                  train_y,  
                  epochs=self.epochs_num,
                  validation_data=(test_x, test_y))
        os.makedirs(os.path.join(self.meta_model_folder), exist_ok=True)
        model.save(os.path.join(self.meta_model_folder, 'saved_model'))

    def evaluate_meta_model(self, test, use_tail = False, recreate_model = False):
        predict_probs, predict_labels, real_y  = self.predict_with_meta_model(test,
            with_head_tail=use_tail, recreate_model = recreate_model)
        print(confusion_matrix(real_y, predict_labels))
        print(classification_report(real_y, predict_labels))

    def predict_with_meta_model(self, df, with_head_tail = False, recreate_model = False):
        data = self.prepare_dataset_for_train(df, use_tail = with_head_tail)
        model = tf.keras.models.load_model(os.path.join(self.meta_model_folder, 'saved_model'))
        predict_probs = model.predict(data)
        real_y = [-1]*len(df)
        if self.label_column in df.columns:
            real_y = df[self.label_column].values
        if self.multilabel:
            predicted_labels = np.asarray(predict_probs >= 0.5, dtype=int)
            return predict_probs, predicted_labels, real_y
        else:
            softmax_results = softmax(predict_probs, axis=1)
            predicted_labels = np.argmax(softmax_results, axis=1)
            return softmax_results, predicted_labels, real_y

    def getPredictions(self, in_sentences, get_probabilities = False, is_head = True, with_output_layer=False, window_agg = False):
        label_ids = [0]*len(self.label_list) if self.multilabel else 0
        if not window_agg:
            input_examples = [run_classifier.InputExample(guid="",
                                                          text_a = self.get_text_a_from_tuple(x),
                                                          text_b = self.get_text_b_from_tuple(x),
                                                          label = label_ids) for x in in_sentences]
            input_features = self.convert_examples_to_features(input_examples, self.label_list,
                self.max_seq_length , self.tokenizer, is_head = is_head)
            predict_input_fn = self.input_fn_builder(features=input_features,
                seq_length=self.max_seq_length , is_training=False, drop_remainder=False)
            predictions = self.estimator.predict(predict_input_fn)
            return self.prepare_final_predictions(predictions, with_output_layer, get_probabilities)
        else:
            index2article = {}
            input_examples = []
            index_id = 0
            for idx, x in enumerate(in_sentences):
                text_a = self.get_text_a_from_tuple(x)
                text_b = self.get_text_b_from_tuple(x)
                words = text_b.split(" ")
                start = 0
                while start < len(words):
                    input_examples.append(run_classifier.InputExample(guid="",
                                                          text_a = text_a,
                                                          text_b = " ".join(words[start:start+256]),
                                                          label = label_ids))
                    start += 128
                    index2article[index_id] = idx
                    index_id += 1
            if len(input_examples) % 2 != 0:
                input_examples.append(input_examples[-1])
                index2article[index_id] = index2article[index_id-1]
                index_id += 1

            input_features = self.convert_examples_to_features(input_examples, self.label_list,
                self.max_seq_length , self.tokenizer, is_head = True)

            predict_input_fn = self.input_fn_builder(features=input_features,
                seq_length=self.max_seq_length , is_training=False, drop_remainder=False)
            predictions = self.estimator.predict(predict_input_fn)

            agg_predictions = []
            art_predictions = {}
            art_output_layers = {}
            for ind, prediction in enumerate(predictions):
                art_id = index2article[ind]
                if art_id not in art_predictions:
                    art_predictions[art_id] = []
                    art_output_layers[art_id] = []
                art_predictions[art_id].append(prediction["probabilities"])
                art_output_layers[art_id].append(prediction['bert_output_layer'])
            for art_id in sorted(art_predictions.keys()):
                art_predictions[art_id] = np.mean(art_predictions[art_id], axis=0)
                art_output_layers[art_id] = np.max(art_output_layers[art_id], axis=0)
                agg_predictions.append((self.softmax(art_predictions[art_id]),
                                        np.argmax(art_predictions[art_id]),
                                        art_output_layers[art_id]))

            return agg_predictions

