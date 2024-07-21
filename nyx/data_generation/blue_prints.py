import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from accelerate import PartialState
from datasets import DatasetDict
from pydantic import BaseModel, Field
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)

from nyx.constants import COMMON_OUTPUT_PATHS, METRICS_PATH, RM_TRAIN_DATA_PATH
from nyx.utils import precision_enumerator


class AbstractController(ABC):
    @abstractmethod
    def label_data(self):
        pass

    @abstractmethod
    def report_on_performance(self):
        pass


class ModelAndTokeniserInConfig(BaseModel):
    llm_model_name: str = Field(examples=['google/flan-t5-small'])
    tokeniser_name: Optional[str] = None
    precision_name: str = Field(examples=['float16'])
    device: str = Field(examples=['mps', 'cuda', 'cpu'])
    dataset: Any  # DatasetDict  # = Field(default_factory=DatasetDict)
    run_id: str = Field(examples=[uuid.uuid4().hex])


class AbstractDataGenerator(ABC):
    def __init__(self, config: Dict[str, Any], multi_gpu_setup: bool = False):
        """Abstract class for data generators to set up all the common requirements for the data generation task.
        Notes
        -----
        The input dictionary is validated as ModelAndTokeniserInConfig
        """
        self.config = config
        self.validate_config()
        self.multi_gpu_setup = multi_gpu_setup

        self.gpu_type = (
            torch.cuda.get_device_name()
            if multi_gpu_setup is True
            else config.get('device')
        )
        self.n_gpus_available = 1
        self.duration = 0
        # https://stackoverflow.com/questions/1639174/creating-class-instance-properties-from-a-dictionary
        for key, value in config.items():
            setattr(self, key, value)

        # self.validate_run_id()

        self.precision = precision_enumerator(self.precision_name)

        if multi_gpu_setup is True:
            self.distributed_state = PartialState()
        access_token = os.environ.get('HF_TOKEN')
        try:
            self.labeller_model = (
                AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    device_map=self.distributed_state.device,
                    # attn_implementation="flash_attention_2",
                    token=access_token,
                )
                if self.multi_gpu_setup is True
                else AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    token=access_token,
                    # attn_implementation="flash_attention_2",
                ).to(torch.device(self.device))
            )

        except ValueError:
            self.labeller_model = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    device_map=self.distributed_state.device,
                    # attn_implementation="flash_attention_2",
                    token=access_token,
                )
                if self.multi_gpu_setup is True
                else AutoModelForSeq2SeqLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    token=access_token,
                    #                     attn_implementation="flash_attention_2",
                ).to(torch.device(self.device))
            )

        self.tokeniser_name = (
            self.tokeniser_name
            if self.tokeniser_name is not None
            else self.llm_model_name
        )
        # Tokeniser padding should be left, so that the right most token is the most recent token. Thus making it easier
        # to get the right logits for probability computations.
        self.padding = 'left'
        self.tokeniser = AutoTokenizer.from_pretrained(
            self.tokeniser_name, padding_side=self.padding, token=access_token
        )
        self.tokeniser.pad_token = (
            self.tokeniser.pad_token
            if self.tokeniser.pad_token is not None
            else self.tokeniser.eos_token
        )

    @abstractmethod
    def generate_labels(self):
        # all the different prompt methods can go in here, which can then be uniformly called from the controller.
        pass

    @abstractmethod
    def validate_config(self):
        """This method ensures all required parameters are passed and are valid. Furthermore, this method adds all the
        optional parameters to the config."""
        config_model = ModelAndTokeniserInConfig(**self.config)
        self.config = dict(config_model)

    def save_rm_training_data(self, dataset):
        train_dataset_dict = DatasetDict({"train": dataset})

        common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=self.run_id)
        rm_train_data_path = RM_TRAIN_DATA_PATH.format(COMMON_OUTPUT_PATHS=common_path)
        train_dataset_dict.save_to_disk(rm_train_data_path)
        print(f"Successfully saved data to:\n{rm_train_data_path}")

        return train_dataset_dict


class AbstractEvaluator(ABC):
    def __init__(self, data_to_evaluate: DatasetDict, run_id: str):
        self.data_to_evaluate = data_to_evaluate
        self.run_id = run_id
        self.metrics = dict()

    def log_metrics(self, predicted_col: str = 'ai_choice'):
        common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=self.run_id)
        data_path = METRICS_PATH.format(COMMON_OUTPUT_PATHS=common_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if predicted_col == 'ai_choice':
            datapath = f'{data_path}/labeller-results.json'
        elif predicted_col == 'rm_choice':
            datapath = f'{data_path}/reward-model-results.json'
        else:
            raise NotImplementedError(
                'Predicted column must be either "ai_choice" or "rm_choice", to evaluate the '
                'performance of either the labelling method or the reward model respectively.'
            )
        with open(datapath, 'w') as file:
            json.dump(self.metrics, file)

    @abstractmethod
    def compute_metrics(self):
        pass
