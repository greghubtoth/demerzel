import numpy as np
from datasets import DatasetDict
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef)

from nyx.data_generation.blue_prints import AbstractEvaluator


class AILabelEvaluator(AbstractEvaluator):
    def __init__(self, data_to_evaluate: DatasetDict, run_id: str):
        super().__init__(data_to_evaluate, run_id)

    def compute_labeller_alignment(
        self, data_split: str = 'train', predicted_col: str = 'ai_choice'
    ):
        def compare_features(example):
            example["is_match"] = (
                1 if example["choice"] == example[predicted_col] else 0
            )
            return example

        # Apply the function to each example in the dataset
        if data_split == 'train':
            self.data_to_evaluate = self.data_to_evaluate.map(compare_features)
        else:  # data_split == 'test':
            self.data_to_evaluate[data_split] = self.data_to_evaluate[data_split].map(
                compare_features
            )

        # print(self.data_to_evaluate)
        # Calculate the mean value of the 'is_match' feature
        instruct_feedback_agreement_mean_value = np.round(
            np.mean(self.data_to_evaluate[data_split]["is_match"]) * 100, 2
        )
        print(
            "The prompt driven, instruct model generated feedback labels are in agreement with the annotator provided"
            f" labels: {instruct_feedback_agreement_mean_value}% of the times."
        )
        metric_name = (
            'labeller_alignment'
            if predicted_col == 'ai_choice'
            else 'pairwise_accuracy'
        )
        self.metrics.update({metric_name: instruct_feedback_agreement_mean_value})

    def compute_classification_report(
        self, data_split: str = 'train', predicted_col: str = 'ai_choice'
    ):
        cr_results_str = classification_report(
            y_true=self.data_to_evaluate[data_split]["choice"],
            y_pred=self.data_to_evaluate[data_split][predicted_col],
            output_dict=False,
        )
        cr_results_dict = classification_report(
            y_true=self.data_to_evaluate[data_split]["choice"],
            y_pred=self.data_to_evaluate[data_split][predicted_col],
            output_dict=True,
        )
        print(cr_results_str)
        self.metrics.update({'classification_report': cr_results_dict})

    def compute_confusion_matrix(
        self, data_split: str = 'train', predicted_col: str = 'ai_choice'
    ):
        cm_results = confusion_matrix(
            y_true=self.data_to_evaluate[data_split]["choice"],
            y_pred=self.data_to_evaluate[data_split][predicted_col],
        )
        tn, fp, fn, tp = cm_results.ravel()
        print(f'tp: {tp}, fp: {fp}\nfn: {fn}, tn: {tn}')
        self.metrics.update(
            {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        )

    def compute_mcc(self, data_split: str = 'train', predicted_col: str = 'ai_choice'):
        mcc = matthews_corrcoef(
            y_true=self.data_to_evaluate[data_split]["choice"],
            y_pred=self.data_to_evaluate[data_split][predicted_col],
        )
        print(f'MCC is {round(mcc, 4)}.')
        self.metrics.update({'mcc': mcc})

    def compute_metrics(
        self, data_split: str = 'train', predicted_col: str = 'ai_choice'
    ):
        self.compute_labeller_alignment(
            data_split=data_split, predicted_col=predicted_col
        )
        self.compute_classification_report(
            data_split=data_split, predicted_col=predicted_col
        )
        self.compute_confusion_matrix(
            data_split=data_split, predicted_col=predicted_col
        )
        self.compute_mcc(data_split=data_split, predicted_col=predicted_col)
        self.log_metrics(predicted_col=predicted_col)
