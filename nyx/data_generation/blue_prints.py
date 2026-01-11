import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from accelerate import PartialState
from datasets import DatasetDict
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from nyx.constants import COMMON_OUTPUT_PATHS, METRICS_PATH, RM_TRAIN_DATA_PATH
from nyx.utils import precision_enumerator

# from unsloth import FastLanguageModel
# from accelerate.utils import BnbQuantizationConfig


class AbstractController(ABC):
    @abstractmethod
    def label_data(self):
        pass

    @abstractmethod
    def report_on_performance(self):
        pass


class ModelAndTokeniserInConfig(BaseModel):
    llm_model_name: str = Field(examples=["google/flan-t5-small"])
    tokeniser_name: Optional[str] = None
    precision_name: str = Field(examples=["float16"])
    device: str = Field(examples=["mps", "cuda", "cpu"])
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
        # print(f'In the abstract data generator the config is: {self.config}.')
        self.validate_config()
        self.multi_gpu_setup = multi_gpu_setup

        self.gpu_type = (
            torch.cuda.get_device_name()
            if multi_gpu_setup is True
            else config.get("device")
        )
        self.n_gpus_available = 1
        self.duration = 0
        # https://stackoverflow.com/questions/1639174/creating-class-instance-properties-from-a-dictionary
        for key, value in config.items():
            # print(f'assigning values: {key}={value} to abstract data generator.')
            setattr(self, key, value)

        # self.validate_run_id()

        self.precision = precision_enumerator(self.precision_name)

        if multi_gpu_setup is True:
            self.distributed_state = PartialState()
        access_token = os.environ.get("HF_TOKEN")

        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = (
            True  # Use 4bit quantization to reduce memory usage. Can be False.
        )

        try:
            # bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            #                                                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            self.labeller_model = (
                AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    device_map=self.distributed_state.device,
                    # attn_implementation="flash_attention_2",
                    token=access_token,
                    # load_in_4bit=load_in_4bit, # works on its own, but will be deprecated
                )
                if self.multi_gpu_setup is True
                else AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=self.precision,
                    token=access_token,
                    # attn_implementation="flash_attention_2",
                ).to(torch.device(self.device))
            )
            # self.labeller_model, self.tokeniser = FastLanguageModel.from_pretrained(
            #     model_name=self.llm_model_name,  # Reminder we support ANY Hugging Face model!
            #     max_seq_length=8_000,
            #     dtype=dtype,
            #     load_in_4bit=load_in_4bit,
            #     device_map=self.distributed_state.device,
            #     # token=access_token,
            #     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
            # )
            # self.tokeniser.padding_side = 'left'
            # FastLanguageModel.for_inference(self.labeller_model)

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
        self.padding = "left"
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

    def log_metrics(self, predicted_col: str = "ai_choice"):
        common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=self.run_id)
        data_path = METRICS_PATH.format(COMMON_OUTPUT_PATHS=common_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if predicted_col == "ai_choice":
            datapath = f"{data_path}/labeller-results.json"
        elif predicted_col == "rm_choice":
            datapath = f"{data_path}/reward-model-results.json"
        else:
            raise NotImplementedError(
                'Predicted column must be either "ai_choice" or "rm_choice", to evaluate the '
                "performance of either the labelling method or the reward model respectively."
            )
        with open(datapath, "w") as file:
            json.dump(self.metrics, file)

    @abstractmethod
    def compute_metrics(self):
        pass


# ============================================================================
# vLLM-Based Abstract Data Generator
# ============================================================================


class VLLMConfig(BaseModel):
    """Configuration for vLLM backend"""

    max_model_len: Optional[int] = 4096
    enable_lora: bool = False
    lora_adapter_path: Optional[str] = None
    enable_prefix_caching: bool = True
    quantization: Optional[str] = None  # 'awq', 'gptq', 'fp8', 'bitsandbytes'
    load_format: Optional[str] = None  # 'auto', 'bitsandbytes'
    gpu_memory_utilization: float = 0.85
    # make the below two parameters optional
    tensor_parallel_size: Optional[int | None] = (
        None  # Number of GPUs for tensor parallelism
    )
    pipeline_parallel_size: Optional[int | None] = (
        None  # Number of nodes for pipeline parallelism
    )
    temperature: float = 0.0
    top_k: int = 1
    max_tokens: int = 512
    trust_remote_code: bool = True
    logprobs: int = 5  # Number of top logprobs to return per token
    # max_logprobs: int = 20  # Maximum logprobs that can be requested


class AbstractVLLMDataGenerator(ABC):
    """
    Pure vLLM-based data generator for high-performance inference.

    This abstract class provides the foundation for vLLM-based labeling methods,
    handling model initialization, batched generation, and data saving.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - llm_model_name: str - HuggingFace model name or path
        - run_id: str - Unique run identifier
        - dataset: DatasetDict - Dataset to label
        - vllm_config: dict - vLLM-specific configuration (optional)

    Notes
    -----
    Subclasses must implement:
    - generate_labels(): Main labeling logic
    - validate_config(): Configuration validation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()

        # Set attributes from config
        for key, value in config.items():
            if key != "vllm_config":
                setattr(self, key, value)

        # Extract vLLM configuration
        self.vllm_config = VLLMConfig(**config.get("vllm_config", {}))

        # GPU info
        self.n_gpus_available = torch.cuda.device_count()
        self.gpu_type = (
            torch.cuda.get_device_name() if self.n_gpus_available > 0 else "cpu"
        )
        self.duration = 0

        # Initialize vLLM
        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM engine"""
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        print(f"Initializing vLLM for {self.llm_model_name}")
        print(f"Detected {self.n_gpus_available} GPU(s): {self.gpu_type}")

        # Build kwargs dictionary, excluding None values
        llm_kwargs = {
            "model": self.llm_model_name,
            "max_model_len": self.vllm_config.max_model_len,
            "trust_remote_code": self.vllm_config.trust_remote_code,
            "enable_lora": self.vllm_config.enable_lora,
            "enable_prefix_caching": self.vllm_config.enable_prefix_caching,
            "gpu_memory_utilization": self.vllm_config.gpu_memory_utilization,
        }

        # Add optional parameters only if they're not None
        if self.vllm_config.quantization is not None:
            llm_kwargs["quantization"] = self.vllm_config.quantization
        if self.vllm_config.load_format is not None:
            llm_kwargs["load_format"] = self.vllm_config.load_format
        if self.vllm_config.tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = self.vllm_config.tensor_parallel_size
        if self.vllm_config.pipeline_parallel_size is not None:
            llm_kwargs["pipeline_parallel_size"] = (
                self.vllm_config.pipeline_parallel_size
            )

        self.llm = LLM(**llm_kwargs)

        # Initialize tokenizer (needed for logprob parsing)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        self.sampling_params_probabilities = SamplingParams(
            temperature=0,
            top_k=1,
            max_tokens=5,
            logprobs=self.vllm_config.logprobs,  # Request top-k logprobs per token
        )
        self.sampling_params = SamplingParams(
            temperature=self.vllm_config.temperature,
            top_k=self.vllm_config.top_k,
            max_tokens=self.vllm_config.max_tokens,
        )

        print(
            f"vLLM initialized successfully with logprobs={self.vllm_config.logprobs}"
        )

    def generate_batch(self, prompts: List[str], return_logprobs: bool = False):
        """
        Generate text for a batch of prompts using vLLM.

        Parameters
        ----------
        prompts : List[str]
            List of prompts to generate from
        return_logprobs : bool
            If True, return full vLLM outputs with logprobs; if False, return only text

        Returns
        -------
        Union[List[str], List[RequestOutput]]
            If return_logprobs=False: List of generated text strings
            If return_logprobs=True: List of vLLM RequestOutput objects (with logprobs)
        """

        lora_request = None
        if self.vllm_config.enable_lora and self.vllm_config.lora_adapter_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                "labeller_adapter", 1, self.vllm_config.lora_adapter_path
            )

        outputs = self.llm.generate(
            prompts,
            self.sampling_params_probabilities
            if return_logprobs
            else self.sampling_params,
            lora_request=lora_request,
        )

        if return_logprobs:
            # Return full outputs with logprobs
            return outputs
        else:
            # Return only text (backward compatible)
            return [output.outputs[0].text for output in outputs]

    def save_rm_training_data(self, dataset):
        """Save reward model training data"""
        train_dataset_dict = DatasetDict({"train": dataset})

        common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=self.run_id)
        rm_train_data_path = RM_TRAIN_DATA_PATH.format(COMMON_OUTPUT_PATHS=common_path)
        train_dataset_dict.save_to_disk(rm_train_data_path)
        print(f"Successfully saved data to:\n{rm_train_data_path}")

        return train_dataset_dict

    @abstractmethod
    def generate_labels(self) -> DatasetDict:
        """Generate labels for the dataset"""
        pass

    @abstractmethod
    def validate_config(self):
        """Validate configuration"""
        pass
