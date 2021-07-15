from interventions_labeling_lib import intervention_labels
from interventions_labeling_lib import intervention_labeling
from interventions_labeling_lib.intervention_labels import InterventionLabels
import numpy as np

class InterventionsLabeller:

    def __init__(self, interventions_model_folder, word2vec_model_folder, status_logger=None):
        self.status_logger = status_logger
        self.interventions_model_folder = interventions_model_folder
        self.word2vec_model_folder = word2vec_model_folder
        self.intervention_labeler = intervention_labeling.InterventionLabeling(google_models_folder=word2vec_model_folder)
        self.intervention_labeler.load_previous_models(interventions_model_folder)

    def log_percents(self, percent):
        if self.status_logger is not None:
            self.status_logger.update_step_percent(percent)

    def label_df(self, df, narrow_concept_column, broad_concepts_name=None, predicted_label_column="Predicted Label"):
        rows_to_take = []
        all_labels = []
        for i in range(len(df)):
            rows_to_take.append(
                (df[narrow_concept_column].values[i], "intervention" if broad_concepts_name is None else df[broad_concepts_name].values[i]))
            if (i % 1000 == 0 or i == len(df) - 1) and i > 0:
                predicted_labels = [
            InterventionLabels.INTERVENTION_NUMBER_TO_LABEL[label] for label in self.intervention_labeler.predict_class(
                    np.asarray(rows_to_take))]
                rows_to_take = []
                all_labels.extend(predicted_labels)
                self.log_percents(i/len(df)*90)
        df[predicted_label_column] = all_labels
        return df

