from bert_models.base_bert_model import BaseBertModel
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

class BaseBertModelWithBoost(BaseBertModel):

    def __init__(self, output_dir, label_list, boost_model = "", gpu_device_num_hub=0,gpu_device_num = 1, batch_size = 16, max_seq_length = 256,\
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", model_folder = "", label_column = "label",
        use_concat_results=False):
        BaseBertModel.__init__(self, output_dir, label_list, gpu_device_num_hub=gpu_device_num_hub,
            gpu_device_num = gpu_device_num, batch_size = batch_size, max_seq_length = max_seq_length,
            bert_model_hub = bert_model_hub, model_folder = model_folder, label_column = label_column,
            use_concat_results = use_concat_results)
        self.load_boost_model(boost_model)

    def load_boost_model(self, folder):
        self.sgBoost = joblib.load(os.path.join(folder,'GradientBoost.joblib')) if folder != "" else GradientBoostingClassifier()

    def save_boost_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        joblib.dump(self.sgBoost, os.path.join(folder,'GradientBoost.joblib'))

    def prepare_dataset_for_boosting(self, train, use_tail = False):
        es_train_prob_study_type_11, res_train_full_study_type_11, train_y_study_type_11 = self.evaluate_model(train)
        if use_tail:
            es_train_prob_study_type_11_tail, res_train_full_study_type_11_tail, train_y_study_type_11_tail = self.evaluate_model(train, False)
            sg_boost_x = np.concatenate([ es_train_prob_study_type_11, es_train_prob_study_type_11_tail],axis=1)
            return sg_boost_x
        return es_train_prob_study_type_11

    def prepare_datasets_for_boosting(self, train, test, study_df, use_tail = False):
        sg_boost_x = self.prepare_dataset_for_boosting(train, use_tail = use_tail)
        sg_boost_test_x = self.prepare_dataset_for_boosting(test, use_tail = use_tail)
        sg_boost_study_x = self.prepare_dataset_for_boosting(study_df, use_tail = use_tail)
        return sg_boost_x, sg_boost_test_x, sg_boost_study_x

    def train_boost_model(self, train, test, study_df, use_tail = False, n_estimators = 60, max_depth = 8, for_train = True):
        sg_boost_x, sg_boost_test_x, sg_boost_study_x = self.prepare_datasets_for_boosting(train, test, study_df, use_tail = use_tail)
        train_y = list(train[self.label_column].values)
        test_y = list(test[self.label_column].values)
        study_df_y = list(study_df[self.label_column].values)
        if for_train:
            self.sgBoost = GradientBoostingClassifier(n_estimators = n_estimators, max_depth=max_depth)
            self.sgBoost.fit(sg_boost_test_x, test_y)
        print(self.sgBoost.score(sg_boost_test_x, test_y))
        print(self.sgBoost.score(sg_boost_x, train_y))
        print(self.sgBoost.score(sg_boost_study_x, study_df_y))
        print(confusion_matrix(study_df_y, self.sgBoost.predict(sg_boost_study_x)))
        print(classification_report(study_df_y, self.sgBoost.predict(sg_boost_study_x)))

    def evaluate_boost_model(self, test, use_tail = False):
        sg_boost_test_x =self.prepare_dataset_for_boosting(test, use_tail = use_tail)
        test_y = list(test[self.label_column].values)
        print(self.sgBoost.score(sg_boost_test_x, test_y))
        print(confusion_matrix(test_y, self.sgBoost.predict(sg_boost_test_x)))
        print(classification_report(test_y, self.sgBoost.predict(sg_boost_test_x)))

    def predict_with_boosting(self, df, with_head_tail = False):
        res_prob, res_label, res_y = self.predict_for_df(df)
        if with_head_tail:
            res_prob_tail, res_label_tail, res_y = self.predict_for_df(df, is_head = False)
            res_prob = np.concatenate([ res_prob, res_prob_tail],axis=1)
        res_prob, res_label = [self.softmax(x) for x in self.sgBoost.decision_function(res_prob)], self.sgBoost.predict(res_prob)
        return res_prob, res_label, res_y
