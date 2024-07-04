from pathlib import Path

from datasets import load_dataset, load_from_disk

from nyx.constants import COMPARISON_DATA_PATH
from nyx.data_loaders.abstract_data_loader import AbstractDataLoader
from nyx.utils import restructure_and_save_data


class HumanEvaluatedDataLoader(AbstractDataLoader):
    def __init__(self, output_path: str = COMPARISON_DATA_PATH):
        self.output_path = output_path

    def load(self):
        comparison_data_path = Path(self.output_path)
        if not comparison_data_path.is_dir():
            dataset = load_dataset("openai/summarize_from_feedback", name='comparisons')
            print("Restructuring and saving OpenAI labelled reddit summarisation data.")
            restructure_and_save_data(dataset)

        comparison_dataset = load_from_disk(self.output_path)
        return comparison_dataset
