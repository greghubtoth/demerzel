# parallel runnable
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableParallel.html
from typing import Callable

from nyx.data_generation.blue_prints import AbstractController
from nyx.data_generation.evaluators import AILabelEvaluator
from nyx.data_generation.settings import ADAPTED_EXPEL_ET_AL, LABELLING_CLASSES
from nyx.data_loaders import HumanEvaluatedDataLoader


class Controller(AbstractController):
    def __init__(
        self,
        labelling_method,
        labelling_config,
        data_loader=HumanEvaluatedDataLoader,
        evaluator=AILabelEvaluator,
    ):
        self.data_loader = data_loader()
        self.data_to_label = self.data_loader.load()
        self.labelling_method = labelling_method
        self.labelling_config = labelling_config if labelling_config is not None else {}
        self.evaluator = evaluator

    @staticmethod
    def get_labeller(labelling_method: str) -> Callable:
        labelling_func = LABELLING_CLASSES.get(labelling_method)
        if labelling_func is not None:
            return labelling_func
        else:
            raise ValueError(
                f"Only the following classes are supported:\n{LABELLING_CLASSES.keys()}"
            )

    def label_data(self):
        labeller_class = Controller.get_labeller(self.labelling_method)
        self.labelling_config.update({'dataset': self.data_to_label})
        labeller_instance = labeller_class(self.labelling_config)
        self.labelled_data = labeller_instance.generate_labels()
        self.labelling_duration = labeller_instance.duration
        self.n_gpus_available = labeller_instance.n_gpus_available
        self.gpu_type = labeller_instance.gpu_type

        if self.labelling_method == ADAPTED_EXPEL_ET_AL:
            self.labelling_config['insights'] = labeller_instance.insights
        return self.labelled_data

    def report_on_performance(self):
        # Job of the evaluator
        self.evaluator = self.evaluator(
            self.labelled_data, self.labelling_config['run_id']
        )
        self.evaluator.compute_metrics()  # MCC, f1, recall, precision, confusion matrix
        print('Finished generated data evaluation.')
