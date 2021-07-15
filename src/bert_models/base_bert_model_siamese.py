import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

import os
from time import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

class BaseBertModel:
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    def __init__(self, output_dir, label_list, gpu_device_num_hub=0, gpu_device_num = 1, batch_size = 16, max_seq_length = 256,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "", label_column = "label",
        use_concat_results = False):
        if output_dir == "":
            return
        self.gpu_device_num_hub = gpu_device_num_hub
        self.use_concat_results = use_concat_results
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        tf.gfile.MakeDirs(self.output_dir)
        self.bert_model_hub = bert_model_hub
        self.model_folder = model_folder
        self.label_column = label_column
        self.gpu_device_num = gpu_device_num
        self.label_list = label_list
        print("Started tokenizer loading")
        self.tokenizer = self.create_tokenizer_from_hub_module()
        print("Tokenizrer loaded")
        self.create_config([])
        print("Config is done")

    def create_config(self, train_features):
        self.num_train_steps = int(len(train_features) / self.batch_size * BaseBertModel.NUM_TRAIN_EPOCHS)
        self.num_warmup_steps = int(self.num_train_steps * BaseBertModel.WARMUP_PROPORTION)
        self.run_config = tf.estimator.RunConfig(
                model_dir=self.output_dir,
                save_summary_steps=BaseBertModel.SAVE_SUMMARY_STEPS,
                save_checkpoints_steps=BaseBertModel.SAVE_CHECKPOINTS_STEPS)

        if self.model_folder == "":
            self.model_fn = self.model_fn_builder(
              num_labels=len(self.label_list),
              learning_rate=BaseBertModel.LEARNING_RATE,
              num_train_steps=self.num_train_steps,
              num_warmup_steps=self.num_warmup_steps,
              use_concat_results=self.use_concat_results)
        else:
            print(self.model_folder)
            self.bert_config = modeling.BertConfig.from_json_file(os.path.join(self.model_folder,"bert_config.json"))
            self.model_fn = self.model_fn_builder(
              num_labels= len(self.label_list),
              learning_rate=BaseBertModel.LEARNING_RATE,
              num_train_steps=self.num_train_steps,
              num_warmup_steps=self.num_warmup_steps,
              bert_config=self.bert_config,
              init_checkpoint=os.path.join(self.model_folder,"bert_model.ckpt"),
              use_tpu=False,
              use_one_hot_embeddings=False,
              use_concat_results=self.use_concat_results)

        self.estimator = tf.estimator.Estimator(
          model_fn = self.model_fn,
          config=self.run_config,
          params={"batch_size": self.batch_size})

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        with tf.device('/device:GPU:%d'%self.gpu_device_num_hub):
          print("Tokenizer used gpu %d"%self.gpu_device_num_hub)
          if self.model_folder == "":
              with tf.Graph().as_default():
                  bert_module = hub.Module(self.bert_model_hub)
                  tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
                  with tf.Session() as sess:
                      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
              print("Model partly loaded")
              return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
          else:
              dict_path = os.path.join(self.model_folder, 'vocab.txt')
              return tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
          print("Model fully loaded") 

    def get_outputs_from_bert(self, input_ids, input_mask, segment_ids,
        is_predicting, use_one_hot_embeddings=False, bert_config=None):
        if self.model_folder == "":
            bert_module = hub.Module(
                self.bert_model_hub,
                trainable=True)
            bert_inputs = dict(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids)
            bert_outputs = bert_module(
                inputs=bert_inputs,
                signature="tokens",
                as_dict=True)

            # Use "pooled_output" for classification tasks on an entire sentence.
            # Use "sequence_outputs" for token-level output.
            output_layer = bert_outputs["pooled_output"]
        else:
            model = modeling.BertModel(
                config=bert_config,
                is_training=not is_predicting,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            output_layer = model.get_pooled_output()
        return output_layer

    def create_model(self, is_predicting, features, labels,
                 num_labels, bert_config = None, use_one_hot_embeddings = False,
                 use_concat_results=False):
      """Creates a classification model."""
      with tf.device('/device:GPU:%d'%self.gpu_device_num):
          print("Model used for model gpu %d"%self.gpu_device_num)
          output_layer = self.get_outputs_from_bert(features["input_ids"],
            features["input_mask"], features["segment_ids"],
            is_predicting,
            bert_config=bert_config, use_one_hot_embeddings=use_one_hot_embeddings)
          hidden_size = output_layer.shape[-1].value
          print(output_layer.shape)

          if "tail_input_ids" in features:
            tail_output_layer = self.get_outputs_from_bert(features["tail_input_ids"],
            features["tail_input_mask"], features["tail_segment_ids"],
            is_predicting,
            bert_config=bert_config, use_one_hot_embeddings=use_one_hot_embeddings)
            print(tail_output_layer.shape)
          
            output_layer = tf.concat([output_layer, tail_output_layer], axis=-1)
            print(output_layer.shape)
            hidden_size *=2

          # Create our own layer to tune for politeness data.
          output_weights = tf.get_variable(
              "output_weights", [num_labels, hidden_size],
              initializer=tf.truncated_normal_initializer(stddev=0.02))

          output_bias = tf.get_variable(
              "output_bias", [num_labels], initializer=tf.zeros_initializer())

          with tf.variable_scope("loss"):

            # Dropout helps prevent overfitting
            if not is_predicting:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
              return (predicted_labels, log_probs, output_layer)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    def model_fn_builder(self, num_labels, learning_rate, num_train_steps,
                         num_warmup_steps, bert_config=None, init_checkpoint=None,
                         use_one_hot_embeddings=False, use_tpu=False, use_concat_results=False):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        
        # TRAIN and EVAL
        if not is_predicting:

          (loss, predicted_labels, log_probs) = self.create_model(
            is_predicting, features, label_ids, num_labels, 
            bert_config = bert_config, use_one_hot_embeddings = use_one_hot_embeddings,
            use_concat_results=use_concat_results)
          
          if self.model_folder != "":
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                if use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
            with tf.device('/device:GPU:%d'%self.gpu_device_num):
              print("Used for model gpu %d"%self.gpu_device_num)
              train_op = bert.optimization.create_optimizer(
                  loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=use_tpu)

              # Calculate evaluation metrics. 
              def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels) 
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)   
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)  
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

              eval_metrics = metric_fn(label_ids, predicted_labels)

              
              if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  train_op=train_op)
              else:
                  return tf.estimator.EstimatorSpec(mode=mode,
                    loss=loss,
                    eval_metric_ops=eval_metrics)
        else:

          (predicted_labels, log_probs, bert_output_layer) = self.create_model(
            is_predicting, features, label_ids, num_labels,
            bert_config = bert_config, use_one_hot_embeddings = use_one_hot_embeddings)

          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels,
              'bert_output_layer': bert_output_layer
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Return the actual model function in the closure
      return model_fn

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length, is_head = True):
      """Truncates a sequence pair in place to the maximum length."""

      # This is a simple heuristic which will always truncate the longer sequence
      # one token at a time. This makes more sense than truncating an equal percent
      # of tokens from each, since if one sequence is very short then each token
      # that's truncated likely contains more information than a longer sequence.
      if is_head:
        while True:
          total_length = len(tokens_a) + len(tokens_b)
          if total_length <= max_length:
            break
          if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
          else:
            tokens_b.pop()
      else:
        tokens_a_inv = tokens_a[::-1]
        tokens_b_inv = tokens_b[::-1]
        while True:
          total_length = len(tokens_a_inv) + len(tokens_b_inv)
          if total_length <= max_length:
            break
          if len(tokens_a_inv) > len(tokens_b_inv):
            tokens_a_inv.pop()
          else:
            tokens_b_inv.pop()
        tokens_a = tokens_a_inv[::-1]
        tokens_b = tokens_b_inv[::-1]
      return tokens_a, tokens_b

    def convert_single_example(self, ex_index, example, label_list, max_seq_length,
                           tokenizer, is_head = True):
          """Converts a single `InputExample` into a single `InputFeatures`."""

          if isinstance(example, run_classifier.PaddingInputExample):
            return run_classifier.InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

          label_map = {}
          for (i, label) in enumerate(label_list):
            label_map[label] = i

          tokens_a = tokenizer.tokenize(example.text_a)
          tokens_b = None
          if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

          if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3, is_head)
          else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
              tokens_a = tokens_a[0:(max_seq_length - 2)] if is_head else tokens_a[-(max_seq_length - 2):]

          # The convention in BERT is:
          # (a) For sequence pairs:
          #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
          #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
          # (b) For single sequences:
          #  tokens:   [CLS] the dog is hairy . [SEP]
          #  type_ids: 0     0   0   0  0     0 0
          #
          # Where "type_ids" are used to indicate whether this is the first
          # sequence or the second sequence. The embedding vectors for `type=0` and
          # `type=1` were learned during pre-training and are added to the wordpiece
          # embedding vector (and position vector). This is not *strictly* necessary
          # since the [SEP] token unambiguously separates the sequences, but it makes
          # it easier for the model to learn the concept of sequences.
          #
          # For classification tasks, the first vector (corresponding to [CLS]) is
          # used as the "sentence vector". Note that this only makes sense because
          # the entire model is fine-tuned.
          tokens = []
          segment_ids = []
          tokens.append("[CLS]")
          segment_ids.append(0)
          for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
          tokens.append("[SEP]")
          segment_ids.append(0)

          if tokens_b:
            for token in tokens_b:
              tokens.append(token)
              segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

          input_ids = tokenizer.convert_tokens_to_ids(tokens)

          # The mask has 1 for real tokens and 0 for padding tokens. Only real
          # tokens are attended to.
          input_mask = [1] * len(input_ids)

          # Zero-pad up to the sequence length.
          while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

          assert len(input_ids) == max_seq_length
          assert len(input_mask) == max_seq_length
          assert len(segment_ids) == max_seq_length

          label_id = label_map[example.label]
          if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

          feature = run_classifier.InputFeatures(
              input_ids=input_ids,
              input_mask=input_mask,
              segment_ids=segment_ids,
              label_id=label_id,
              is_real_example=True)
          return feature

    def convert_examples_to_features(self, examples, label_list, max_seq_length,
                                 tokenizer, is_head = True):
      """Convert a set of `InputExample`s to a list of `InputFeatures`."""

      features = []
      for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
          tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = self.convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, is_head)

        features.append(feature)
      return features

    def prepare_input_features(self, df, is_head = True):
        train_InputExamples = df.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a =  self.get_text_a(x),
                                                                   text_b = self.get_text_b(x),
                                                                   label = x[self.label_column]), axis = 1)
        train_features = self.convert_examples_to_features(train_InputExamples, self.label_list, self.max_seq_length,
          self.tokenizer, is_head = is_head)
        return train_features

    def get_text_a(self, x):
        return ""

    def get_text_b(self, x):
        return ""


    def input_fn_builder(self, features, tail_features, seq_length, is_training, drop_remainder):
      """Creates an `input_fn` closure to be passed to TPUEstimator."""

      all_input_ids = []
      all_input_mask = []
      all_segment_ids = []
      all_label_ids = []
      tail_input_ids = []
      tail_input_mask = []
      tail_segment_ids = []

      for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)
      for feature in tail_features:
        tail_input_ids.append(feature.input_ids)
        tail_input_mask.append(feature.input_mask)
        tail_segment_ids.append(feature.segment_ids)

      def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "tail_input_ids":
                tf.constant(
                    tail_input_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "tail_input_mask":
                tf.constant(
                    tail_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "tail_segment_ids":
                tf.constant(
                    tail_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

      return input_fn

    def train_model(self, train, test = [], use_tails = False):
        train_features = self.prepare_input_features(train)
        tail_features = self.prepare_input_features(train, is_head = False)
        self.create_config(train_features)
        train_input_fn = self.input_fn_builder(
            features=train_features,
            tail_features = tail_features,
            seq_length=self.max_seq_length ,
            is_training=True,
            drop_remainder=False)
        print(f'Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)
        if len(test) > 0:
            return self.evaluate_model(test)

    def predict_for_df(self, train, is_head = True, with_output_layer=False):
        train_vals = list(train.values)
        if self.label_column in train.columns:
            train_y = list(train[self.label_column].values)
        else:
            train_y = [-1]*len(train)
        if len(train_vals) % self.batch_size == 1:
            train_vals.append(train_vals[-1])
            train_y.append(train[self.label_column].values[-1] if self.label_column in train.columns else -1)
        res_train = self.getPredictions(train_vals, get_probabilities = True, is_head = is_head, with_output_layer=with_output_layer)
        res_train_prob = [val[0] for val in res_train]
        res_train_full = [val[1] for val in res_train]
        if with_output_layer:
          return res_train_prob[:len(train)], res_train_full[:len(train)], train_y[:len(train)], [val[2] for val in res_train][:len(train)]
        return res_train_prob[:len(train)], res_train_full[:len(train)], train_y[:len(train)]

    def print_summary(self, test_y, res):
        print(confusion_matrix(test_y, res))
        print(classification_report(test_y, res))
        print("F1 score: ",f1_score(test_y, res, average="macro"))

    def evaluate_model(self, test, is_head = True):
        res_prob, res_label, res_y = self.predict_for_df(test, is_head = is_head)
        self.print_summary(res_y, res_label)
        return res_prob, res_label, res_y

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def getPredictions(self, in_sentences, get_probabilities = False, is_head = True, with_output_layer=False):
        input_examples = [run_classifier.InputExample(guid="", text_a = self.get_text_a_from_tuple(x), text_b = self.get_text_b_from_tuple(x), label = 0) for x in in_sentences] 
        input_features = self.convert_examples_to_features(input_examples, self.label_list, self.max_seq_length , self.tokenizer, is_head = True)
        tail_features = self.convert_examples_to_features(input_examples, self.label_list, self.max_seq_length , self.tokenizer, is_head = False)
        predict_input_fn = self.input_fn_builder(features=input_features,
          tail_features = tail_features,
          seq_length=self.max_seq_length , is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)
        result = []
        for prediction in predictions:
          if with_output_layer:
            result.append((self.softmax(prediction["probabilities"]), prediction['labels'], prediction['bert_output_layer']))
          elif get_probabilities:
            result.append((self.softmax(prediction["probabilities"]), prediction['labels']))
          else:
            result.append(prediction['labels'])

        return result

    def get_text_a_from_tuple(self, x):
        return ""

    def get_text_b_from_tuple(self, x):
        return ""