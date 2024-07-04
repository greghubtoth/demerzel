import functools
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import weaviate
from datasets import Dataset, DatasetDict, concatenate_datasets
from deprecated import deprecated
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate

from nyx.constants import COMMON_OUTPUT_PATHS, PROMPTS_COL, RM_TRAIN_DATA_PATH
from nyx.data_generation.blue_prints import (AbstractDataGenerator,
                                             ModelAndTokeniserInConfig)
from nyx.data_generation.utils import (
    generate_cot_for_prompts_with_gpus, generate_insights_successful,
    generate_insights_with_comparisons, generate_next_token_probabilities_gpus,
    generate_reflexion_and_cot_completions_with_gpus,
    generate_unbiased_ai_preference_distribution, get_documents_from_data,
    get_mean_of_probabilities, prompt_generator_Lee_et_al)


class BaselineLeeEtAlConfigValidator(ModelAndTokeniserInConfig):
    max_new_tokens: int = 256
    # Adjust this based on computational resources
    batch_size: Optional[int] = 2
    target_words: Optional[List[str]] = None
    prompt_col: Optional[str] = PROMPTS_COL


@deprecated(
    version="0.2.0",
    reason="This function is only suitable for single GPU inference. For multi GPU inference "
    "use BaselineLeeEtAlDataGeneratorWithLangChain.",
)
class BaselineLeeEtAlDataGenerator(AbstractDataGenerator):

    def __init__(self, config: Dict[str, Any]):
        """
        Notes
        -----
        The config passed is validated as BaselineLeeEtAlConfigValidator. In other words, see the Validator for required
        and optional arguments."""
        self.config = config
        self.validate_config()
        # self.validate_run_id()
        super().__init__(self.config)

    def generate_labels(self) -> DatasetDict:
        prompt_assembled_datasets = self.dataset.map(
            prompt_generator_Lee_et_al, batched=True
        )
        reversed_prompt_assembled_datasets = self.dataset.map(
            functools.partial(prompt_generator_Lee_et_al, reverse=True), batched=True
        )

        start = time.time()
        ai_choice_list = generate_unbiased_ai_preference_distribution(
            datasets=[prompt_assembled_datasets, reversed_prompt_assembled_datasets],
            original_model=self.labeller_model,
            tokeniser=self.tokeniser,
            batch_size=self.batch_size,
            target_words=self.target_words,
            device=self.device,
        )
        end = time.time()

        duration = end - start
        print(f"Labelling all data twice took {round(duration, 2)} seconds to execute.")

        comparison_train_dataset = prompt_assembled_datasets["train"].add_column(
            name="ai_choice", column=ai_choice_list
        )
        comparison_train_dataset = comparison_train_dataset.add_column(
            name="constant_col",
            column=[1 for _ in range(len(comparison_train_dataset))],
        )

        train_dataset_dict = DatasetDict({"train": comparison_train_dataset})

        common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=self.run_id)
        rm_train_data_path = RM_TRAIN_DATA_PATH.format(COMMON_OUTPUT_PATHS=common_path)
        train_dataset_dict.save_to_disk(rm_train_data_path)
        print(f"Successfully saved data to:\n{rm_train_data_path}")

        return train_dataset_dict

    def validate_config(self):
        config_model = BaselineLeeEtAlConfigValidator(**self.config)
        self.config = dict(config_model)


class CotGeneratorWithGpus(AbstractDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        """
        Notes
        -----
        The config passed is validated as BaselineLeeEtAlConfigValidator. In other words, see the Validator for required
        and optional arguments.
        """
        self.config = config
        self.validate_config()
        super().__init__(self.config, multi_gpu_setup=True)

    def validate_config(self):
        pass

    def generate_cot_with_gpus(
        self,
        dataset: DatasetDict,
        reverse: bool = False,
        target_words: List[str] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        target_words = target_words if target_words is not None else ["1", "2"]
        cot_generations = generate_cot_for_prompts_with_gpus(
            dataset=dataset,
            labeller_model=self.labeller_model,
            tokeniser=self.tokeniser,
            distributed_state=self.distributed_state,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            reverse=reverse,
        )
        predictions = generate_next_token_probabilities_gpus(
            model=self.labeller_model,
            tokeniser=self.tokeniser,
            decoded_reasoning=cot_generations,
            batch_size=self.batch_size,
            distributed_state=self.distributed_state,
            dtype=self.precision,
            target_words=target_words,
        )
        return cot_generations, predictions

    def add_relevant_columns_to_dataset(
        self,
        dataset_to_add_new_cols,
        ai_predicted_label_list,
        ordered_prompt,
        reversed_prompt,
        add_incorrect_prediction_column: bool = False,
        nth_retry: int = 0,
    ) -> DatasetDict:
        columns_to_add_to_dataset = [
            "ai_choice",
            "constant_col",
            "ordered_prompt_used_to_predict",
            "reversed_prompt_used_to_predict",
            "incorrect_prediction",
            "nth_retry",
        ]

        # Ensuring columns are updated if multiple retries occur.
        if columns_to_add_to_dataset[0] in dataset_to_add_new_cols.column_names:
            dataset_to_add_new_cols = dataset_to_add_new_cols.remove_columns(
                columns_to_add_to_dataset
            )

        comparison_train_dataset = dataset_to_add_new_cols.add_column(
            name="ai_choice", column=ai_predicted_label_list
        )
        comparison_train_dataset = comparison_train_dataset.add_column(
            name="constant_col",
            column=[1 for _ in range(len(comparison_train_dataset))],
        )
        comparison_train_dataset = comparison_train_dataset.add_column(
            name="nth_retry",
            column=[nth_retry for _ in range(len(comparison_train_dataset))],
        )

        comparison_train_dataset = comparison_train_dataset.add_column(
            name="ordered_prompt_used_to_predict", column=ordered_prompt
        )
        comparison_train_dataset = comparison_train_dataset.add_column(
            name="reversed_prompt_used_to_predict", column=reversed_prompt
        )
        if add_incorrect_prediction_column is True:

            def compare_features(example):
                example["incorrect_prediction"] = (
                    "True" if example["choice"] != example["ai_choice"] else "False"
                )
                return example

            comparison_train_dataset = comparison_train_dataset.map(compare_features)

        return comparison_train_dataset


class BaselineLeeEtAlDataGeneratorWithLangChain(CotGeneratorWithGpus):

    def __init__(self, config: Dict[str, Any]):
        """
        Notes
        -----
        The config passed is validated as BaselineLeeEtAlConfigValidator. In other words, see the Validator for required
        and optional arguments.
        """
        self.config = config
        self.validate_config()
        super().__init__(self.config)

        # self.pipe = pipeline("text-generation", model=self.labeller_model,
        #                      tokenizer=self.tokeniser,
        #                      max_new_tokens=self.max_new_tokens,
        #                      )  # device=self.device
        # Be mindful that hf_pipeline only returns the prompt and response as a string.
        # self.hf_piped_llm = HuggingFacePipeline(pipeline=self.pipe)
        self.n_gpus_available = torch.cuda.device_count()
        print(f"Number of GPUs detected as available is: {self.n_gpus_available}.")

    def generate_labels(self) -> DatasetDict:
        start = time.time()
        target_words = ["1", "2"]
        predictions_with_both_ordered_combinations = {}

        # ordered_reasoning = generate_cot_for_prompts_with_gpus(dataset=self.dataset,
        #                                                        labeller_model=self.labeller_model,
        #                                                        tokeniser=self.tokeniser,
        #                                                        distributed_state=self.distributed_state,
        #                                                        batch_size=self.batch_size,
        #                                                        max_new_tokens=self.max_new_tokens,
        #                                                        reverse=False)
        # predictions_with_both_ordered_combinations[0] = generate_ai_label_predictions_gpus(model=self.labeller_model,
        #                                                                                    tokeniser=self.tokeniser,
        #                                                                                    decoded_reasoning=ordered_reasoning,
        #                                                                                    batch_size=self.batch_size,
        #                                                                                    distributed_state=self.distributed_state,
        #                                                                                    dtype=self.precision,
        #                                                                                    target_words=target_words)

        # reversed_reasoning = generate_cot_for_prompts_with_gpus(dataset=self.dataset,
        #                                                         labeller_model=self.labeller_model,
        #                                                         tokeniser=self.tokeniser,
        #                                                         distributed_state=self.distributed_state,
        #                                                         batch_size=self.batch_size,
        #                                                         max_new_tokens=self.max_new_tokens,
        #                                                         reverse=True)
        # predictions_with_both_ordered_combinations[1] = generate_ai_label_predictions_gpus(model=self.labeller_model,
        #                                                                                    tokeniser=self.tokeniser,
        #                                                                                    decoded_reasoning=reversed_reasoning,
        #                                                                                    batch_size=self.batch_size,
        #                                                                                    distributed_state=self.distributed_state,
        #                                                                                    dtype=self.precision,
        #                                                                                    target_words=target_words)

        ordered_reasoning, predictions_with_both_ordered_combinations[0] = (
            self.generate_cot_with_gpus(
                dataset=self.dataset, reverse=False, target_words=target_words
            )
        )

        reversed_reasoning, predictions_with_both_ordered_combinations[1] = (
            self.generate_cot_with_gpus(
                dataset=self.dataset, reverse=True, target_words=target_words
            )
        )

        ai_choice_list = get_mean_of_probabilities(
            predictions_with_both_ordered_combinations, target_words
        )
        end = time.time()
        self.duration = round(end - start, 2)

        print(f"Labelling all data twice took {self.duration} seconds to execute.")

        comparison_train_dataset = self.add_relevant_columns_to_dataset(
            dataset_to_add_new_cols=self.dataset['train'],
            ai_predicted_label_list=ai_choice_list,
            ordered_prompt=ordered_reasoning,
            reversed_prompt=reversed_reasoning,
        )
        return self.save_rm_training_data(comparison_train_dataset)

    def validate_config(self):
        config_model = BaselineLeeEtAlConfigValidator(**self.config)
        self.config = dict(config_model)


class ExpelAdaptationConfigValidator(ModelAndTokeniserInConfig):
    max_new_tokens: int = 256
    # Adjust this based on computational resources
    batch_size: Optional[int] = 2
    # langchain_batch_size
    target_words: Optional[List[str]] = None
    prompt_col: Optional[str] = PROMPTS_COL
    # Shinn et al. when n_retries >= 1 then Reflexion is generated.
    n_retries: Optional[int] = 1
    # Adapted Zhao et al. To generate insights, if not provided then data will dictate.
    insights_step_size: Optional[int] = None
    # Li et al. Negative examples are saved and can be retrieved for prompts.
    negative_examples: Optional[bool] = False
    embedding_model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2"
    vdb_search_type: Optional[str] = "similarity"
    max_vdb_documents: Optional[int] = 5_000


class ExpelZhaoEtAlAdaptedDataGenerator(CotGeneratorWithGpus):

    def __init__(self, config: Dict[str, Any]):
        """
        Notes
        -----
        The config passed is validated as ExpelAdaptationConfigValidator. In other words, see the Validator for required
        and optional arguments.
        """
        self.doc_ids = []
        self.insights = ''
        self.config = config
        self.validate_config()
        super().__init__(self.config)
        self.n_gpus_available = torch.cuda.device_count()
        print(f"Number of GPUs detected as available is: {self.n_gpus_available}.")

    def set_up_vector_db(self):
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # force_download=True,
        )
        client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(),
        )
        self.vectorstore = Weaviate.from_documents(
            # Documents will be added later, as examples and insights are accumulated.
            [],
            embeddings,
            weaviate_url="http://127.0.0.1:8079",
        )

        self.example_retriever = self.vectorstore.as_retriever(
            search_type=self.vdb_search_type,
            search_kwargs={
                "k": 1,
                "where_filter": {
                    "path": ["type"],
                    "operator": "Equal",
                    "valueString": "example",
                },
            },
        )
        # self.insight_retriever = vectorstore.as_retriever(
        #     search_type=self.vdb_search_type,
        #     search_kwargs={
        #         "k": 5,
        #         "where_filter": {"path": ["type"], "operator": "Equal", "valueString": "insight"},
        #     },
        # )

    def generate_labels(self) -> DatasetDict:
        start = time.time()
        # predictions_with_both_ordered_combinations = {}
        target_words = ["1", "2"]
        comparison_train_dataset = Dataset.from_dict({})
        # TBD with semantic embeddings chain

        for j in range(0, len(self.dataset['train']), self.insights_step_size):
            nth_retry = 0
            dataset_within_step_size = self.dataset['train'].select(
                range(
                    j, min(j + self.insights_step_size, self.dataset['train'].num_rows)
                )
            )
            insight_generation_step_dataset = Dataset.from_dict({})
            while (
                nth_retry <= self.n_retries and dataset_within_step_size.num_rows >= 1
            ):
                predictions_with_both_ordered_combinations = {}
                if nth_retry == 0:
                    # Need to split here if Insights and or examples exist.
                    ordered_reasoning, predictions_with_both_ordered_combinations[0] = (
                        self.generate_cot_with_gpus(
                            dataset=dataset_within_step_size,
                            reverse=False,
                            target_words=target_words,
                        )
                    )

                    (
                        reversed_reasoning,
                        predictions_with_both_ordered_combinations[1],
                    ) = self.generate_cot_with_gpus(
                        dataset=dataset_within_step_size,
                        reverse=True,
                        target_words=target_words,
                    )

                elif nth_retry >= 1:

                    # Need to add requested columns in here, to use both ordered_prompt and reversed_prompt!
                    ordered_reasoning, predictions_with_both_ordered_combinations[0] = (
                        self.generate_reflexion_and_cot_with_gpus(
                            dataset=dataset_within_step_size,
                            reverse=False,
                            target_words=target_words,
                        )
                    )

                    (
                        reversed_reasoning,
                        predictions_with_both_ordered_combinations[1],
                    ) = self.generate_reflexion_and_cot_with_gpus(
                        dataset=dataset_within_step_size,
                        reverse=True,
                        target_words=target_words,
                    )

                ai_choice_list = get_mean_of_probabilities(
                    predictions_with_both_ordered_combinations, target_words
                )

                dataset_within_step_size = self.add_relevant_columns_to_dataset(
                    dataset_to_add_new_cols=dataset_within_step_size,
                    ai_predicted_label_list=ai_choice_list,
                    ordered_prompt=ordered_reasoning,
                    reversed_prompt=reversed_reasoning,
                    nth_retry=nth_retry,
                )

                # always union / append the correctly classified dataset
                insight_generation_step_dataset = concatenate_datasets(
                    [
                        insight_generation_step_dataset,
                        dataset_within_step_size.filter(
                            lambda example: example["incorrect_prediction"].startswith(
                                "False"
                            )
                        ),
                    ]
                )

                dataset_within_step_size = dataset_within_step_size.filter(
                    lambda example: example["incorrect_prediction"].startswith("True")
                )

                nth_retry += 1
            # At the end of the while loop, append the negative data too.
            insight_generation_step_dataset = concatenate_datasets(
                [insight_generation_step_dataset, dataset_within_step_size]
            )
            comparison_train_dataset = concatenate_datasets(
                [insight_generation_step_dataset, dataset_within_step_size]
            )

            successful_attempts = insight_generation_step_dataset.filter(
                lambda example: example["incorrect_prediction"].startswith("False")
            )
            self.generate_insights(successful_attempts_dataset=successful_attempts)
            self.add_examples_to_vector_db(dataset=insight_generation_step_dataset)

        end = time.time()
        self.duration = round(end - start, 2)

        print(f"Labelling all data twice took {self.duration} seconds to execute.")

        return self.save_rm_training_data(comparison_train_dataset)

    def generate_insights(
        self, successful_attempts_dataset: DatasetDict, reverse: bool = False
    ):
        self.insights = generate_insights_successful(
            dataset=successful_attempts_dataset,
            labeller_model=self.labeller_model,
            tokeniser=self.tokeniser,
            distributed_state=self.distributed_state,
            insights=self.insights,
            reverse=reverse,
            # max_new_tokens=self.max_new_tokens
        )
        # The below will compare trials that succeeded after a reflexion, so that there is a fail and success for the
        # same labelling exercise.
        self.insights = generate_insights_with_comparisons(
            dataset=successful_attempts_dataset.filter(
                lambda example: example["nth_retry"] >= 1
            ),
            labeller_model=self.labeller_model,
            tokeniser=self.tokeniser,
            distributed_state=self.distributed_state,
            insights=self.insights,
            reverse=reverse,
            # max_new_tokens=self.max_new_tokens
        )

    def add_examples_to_vector_db(self, dataset: DatasetDict, reverse: bool = False):
        doc_ids_added = []
        successful_attempts = dataset.filter(
            lambda example: example["incorrect_prediction"].startswith("False")
        )
        failed_attempts = dataset.filter(
            lambda example: example["incorrect_prediction"].startswith("True")
        )

        if self.negative_examples is True:
            negative_docs = get_documents_from_data(
                failed_attempts,
                negative_examples=True,
                reverse=reverse,
            )
            negative_ids = self.example_retriever.add_documents(documents=negative_docs)
            doc_ids_added.extend(negative_ids)

        positive_docs = get_documents_from_data(
            successful_attempts, negative_examples=False
        )
        positive_ids = self.example_retriever.add_documents(documents=positive_docs)
        doc_ids_added.extend(positive_ids)

        self.doc_ids.extend(doc_ids_added)
        if len(self.doc_ids) > self.max_vdb_documents:
            n_docs_added = len(doc_ids_added)
            docs_to_remove = self.doc_ids[:n_docs_added]
            self.doc_ids = self.doc_ids[n_docs_added:]
            self.vectorstore.delete(docs_to_remove)

    def generate_reflexion_and_cot_with_gpus(
        self,
        dataset: DatasetDict,
        reverse: bool = False,
        target_words: List[str] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        target_words = target_words if target_words is not None else ["1", "2"]
        # If the output is incorrect both the ordered and reversed prompts will be reflected upon.
        # Further optimisation could be to save the token probabilities to score less prompts if at least one is good.
        col_to_predict = (
            'reversed_prompt_used_to_predict'
            if reverse is True
            else 'ordered_prompt_used_to_predict'
        )
        cot_generations = generate_reflexion_and_cot_completions_with_gpus(
            dataset=dataset,
            labeller_model=self.labeller_model,
            tokeniser=self.tokeniser,
            distributed_state=self.distributed_state,
            prompt_col=col_to_predict,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
        )

        predictions = generate_next_token_probabilities_gpus(
            model=self.labeller_model,
            tokeniser=self.tokeniser,
            decoded_reasoning=cot_generations,
            batch_size=self.batch_size,
            distributed_state=self.distributed_state,
            dtype=self.precision,
            target_words=target_words,
        )
        return cot_generations, predictions

    def validate_config(self):
        config_model = ExpelAdaptationConfigValidator(**self.config)
        self.config = dict(config_model)