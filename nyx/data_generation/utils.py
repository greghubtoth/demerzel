import re
from enum import Enum
from operator import itemgetter
from typing import Dict, List

import datasets
import torch
from accelerate.utils import gather_object
from datasets import DatasetDict
from deprecated import deprecated
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm
from transformers import GenerationConfig

from nyx.constants import CANDIDATE_COL, POST_COL, PROMPTS_COL
from nyx.data_generation.prompts import (ENDING_LEE_ET_AL, OPENAI_PREAMBLE,
                                         TASK_WITH_COT_LEE_ET_AL)
from nyx.data_generation.prompts.insights import (
    ALL_SUCCESSES_INSIGHTS_TEMPLATE, FAIL_SUCCESS_COMPARISON_INSIGHTS_TEMPLATE)
from nyx.data_generation.prompts.model_specific_tokens import (BOS_USER_TOKEN,
                                                               EOS_TOKEN)
from nyx.data_generation.prompts.openai_preamble_with_cot import (
    COT_EXAMPLE, INSIGHTS, RATIONALES_SPLIT_STRING, RETRIEVED_EXAMPLE_TEMPLATE)
from nyx.data_generation.prompts.reflection import \
    SUMMARISATION_REFLEXION_PROMPT


@deprecated(
    version="0.2.0",
    reason="This function is only suitable for single GPU inference. For multi GPU inference "
    "see functions in BaselineLeeEtAlDataGeneratorWithLangChain.",
)
def prompt_generator_Lee_et_al(example, reverse: bool = False):
    """Function to create the prompts for labelling.


    Notes
    -----
    Note this function could be used to generate the labels alone, but if CoT or equivalent methods are used then the
    below method, generate_ai_labels_for_data(), contains the end of the prompt for labelling. Therefore, please update
    these functions accordingly when experimenting with data generation techniques.

    Parameters
    ----------
    example : dict
        A dictionary containing the data for labelling.
    reverse : bool, optional
        Reverse variable swaps the orders of the provided summaries to minimise ordering and distance bias in LLMs when
        predicting preferences, by default False.
    """
    prompt = []
    # print(example)
    for values in zip(
        example[POST_COL], example[f"{CANDIDATE_COL}_1"], example[f"{CANDIDATE_COL}_2"]
    ):
        examples = (
            TASK_WITH_COT_LEE_ET_AL.format(
                text=values[0], summary1=values[2], summary2=values[1]
            )
            if reverse is True
            else TASK_WITH_COT_LEE_ET_AL.format(
                text=values[0], summary1=values[1], summary2=values[2]
            )
        )
        prompt.append(OPENAI_PREAMBLE + examples)

    example["prompts"] = prompt
    # example["labels"] = example["choice"]
    return example


@deprecated(
    version="0.2.0",
    reason="This function is only suitable for single GPU inference. For multi GPU inference "
    "see functions in BaselineLeeEtAlDataGeneratorWithLangChain.",
)
def calculate_target_token_probabilities(
    original_model,
    tokeniser,
    encoded_text,
    summary_predictions=None,
    target_words=None,
    multi_gpu_setup=False,
    dtype=torch.float16,
):
    """Method to calculate the target token probabilities.

    Notes
    -----
    This method will likely be called in a loop, based on the batch size of the calling function.

    Parameters
    ----------
    original_model : Causal or Seq2Seq model
        The original model to calculate the target token probabilities with.
    tokeniser : Tokenizer
        Tokeniser used to decode and encode the input text.
    encoded_text : transformers.tokenization_utils_base.BatchEncoding
        Prompts encoded with a tokeniser, e.g. tokeniser(list_of_strings, padding="max_length", return_tensors="pt").
    target_words: List[str]
        The target words, by default ['1', '2'].
    summary_predictions: List[List[float]], optional
        All the normalised predictions to be appended to a list of lists,
        e.g., summary_predictions = [[] for _ in target_words]. Summary predictions are used for single GPU inference.
        For multi_gpu inference summary predictions only exist within the scope of the function.
    dtype: torch.dtype, optional
        The dtype of the target token probabilities, by default torch.float16.

    Returns
    -------
    summary_predictions: List[List[float]]
        All the normalised predictions appended to a list of lists. While lists are mutable, making this list a required
        parameter ensures it exists outside this function, so that it can be appended to for all the batched prompts.
    """
    target_words = target_words if target_words is not None else ["1", "2"]
    summary_predictions = (
        summary_predictions
        if summary_predictions is not None
        else [[] for _ in target_words]
    )

    # Get model predictions for the next token
    prediction_params = (
        dict(input_ids=encoded_text.input_ids)
        if 'causal' in original_model.config.architectures[0].lower()
        else dict(
            input_ids=encoded_text.input_ids,
            decoder_input_ids=original_model._shift_right(encoded_text.input_ids),
        )
    )

    with torch.no_grad():
        outputs = original_model(**prediction_params)
        predictions = outputs.logits

    target_token_ids = tokeniser.encode(target_words, add_special_tokens=False)
    # Vectorized computation of probabilities
    target_token_ids_tensor = torch.tensor(
        target_token_ids
    )  # Convert to tensor for batch processing
    log_probabs = torch.log_softmax(
        predictions[:, -1, :], dim=-1, dtype=dtype
    )  # Compute log softmax in one go

    # Extract the log probabilities for the target token IDs
    log_probabs_target = log_probabs[:, target_token_ids_tensor]

    # Convert log probabilities to probabilities
    softmaxed_log_probabs = torch.softmax(log_probabs_target, dim=-1, dtype=dtype)
    softmaxed_log_probabs = torch.transpose(softmaxed_log_probabs, 0, 1).tolist()
    # print(f'softmaxed_log_probabs: {softmaxed_log_probabs}')
    # Populate summary_predictions
    # for k in range(len(target_words)):
    #     optimised_summaries[k].extend(softmaxed_log_probabs[k])
    summary_predictions[0].extend(softmaxed_log_probabs[0])
    summary_predictions[1].extend(softmaxed_log_probabs[1])
    # print(f'optimised_summaries: {optimised_summaries}')

    if multi_gpu_setup is True:
        return summary_predictions[0], summary_predictions[1]
    else:
        return summary_predictions


@deprecated(
    version="0.2.0",
    reason="This function is only suitable for single GPU inference. For multi GPU inference "
    "see functions in BaselineLeeEtAlDataGeneratorWithLangChain.",
)
def generate_ai_labels_for_data(
    dataset_to_label: datasets.dataset_dict.DatasetDict,  # train dataset will be used for labelling.
    original_model,
    tokeniser,
    device: str = None,
    batch_size: int = 2,  # Adjust this based on your computational resources
    target_words: List[str] = None,
    prompt_col: str = PROMPTS_COL,
    max_new_tokens: int = 256,
) -> List[List[float]]:
    target_words = target_words if target_words is not None else ["1", "2"]
    summary_predictions = [[] for _ in target_words]
    # Process in batches
    for i in tqdm(range(0, len(dataset_to_label["train"][prompt_col]), batch_size,)):
        batch_prompts = dataset_to_label["train"][prompt_col][i : i + batch_size]
        input_ids = tokeniser(
            batch_prompts,
            padding=True,
            # truncation=True,
            return_tensors="pt",
        ).input_ids.to(torch.device(device))

        # Parallelised model generation
        # Get reasoning for CoT.
        with torch.no_grad():
            labeller_outputs = original_model.generate(
                input_ids=input_ids,
                generation_config=(
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokeniser.pad_token_id,
                    )
                    if 'causal' in original_model.config.architectures[0].lower()
                    else GenerationConfig(max_new_tokens=max_new_tokens)
                ),
            )
        # Append Ending.
        # decoded_reasoning = [
        #     tokeniser.decode(reasoning_completion, skip_special_tokens=True) + ENDING_LEE_ET_AL
        #     for reasoning_completion in labeller_outputs
        # ]
        decoded_reasoning = [
            prompt + ENDING_LEE_ET_AL
            for prompt in tokeniser.batch_decode(
                labeller_outputs, skip_special_tokens=True
            )
        ]

        encoded_reasoning = tokeniser(
            decoded_reasoning,
            padding=True,
            # truncation=True,
            return_tensors="pt",
        ).to(torch.device(device))
        print('Probabilities being calculated.')
        summary_predictions = calculate_target_token_probabilities(
            original_model=original_model,
            tokeniser=tokeniser,
            encoded_text=encoded_reasoning,
            target_words=target_words,
            summary_predictions=summary_predictions,
        )

    return summary_predictions


@deprecated(
    version="0.2.0",
    reason="This function is only suitable for single GPU inference. For multi GPU inference "
    "see functions in BaselineLeeEtAlDataGeneratorWithLangChain.",
)
def generate_unbiased_ai_preference_distribution(
    datasets: List[datasets.dataset_dict.DatasetDict],
    original_model,
    tokeniser,
    device: str = None,
    batch_size: int = 2,
    target_words: List[str] = None,
    prompt_col: str = PROMPTS_COL,
    max_new_tokens: int = 256,
) -> List[List[float]]:
    target_words = target_words if target_words is not None else ["1", "2"]
    # mitigating position bias when ordering potential <summary> (LLM response) candidates
    predictions_with_both_ordered_combinations = {}

    for i, dataset in enumerate(datasets):
        pm_values_per_target = generate_ai_labels_for_data(
            dataset_to_label=dataset,
            original_model=original_model,
            tokeniser=tokeniser,
            batch_size=batch_size,
            target_words=target_words,
            prompt_col=prompt_col,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        predictions_with_both_ordered_combinations[i] = pm_values_per_target
    # print(f'ordering_combinations: {ordering_combinations}')
    # predictions_with_both_ordered_combinations[i].reverse()
    ai_predictions = get_mean_of_probabilities(
        predictions_dict=predictions_with_both_ordered_combinations,
        target_words=target_words,
    )
    return ai_predictions


def calculate_target_token_probabilities_with_gpus(
    original_model,
    tokeniser,
    distributed_state,
    encoded_text,
    target_words=None,
    dtype=torch.float16,
) -> List[List[float]]:
    """Method to calculate the target token probabilities.

    Notes
    -----
    This method will likely be called in a loop, based on the batch size of the calling function.

    Parameters
    ----------
    original_model : Causal or Seq2Seq model
        The original model to calculate the target token probabilities with.
    tokeniser : Tokenizer
        Tokeniser used to decode and encode the input text.
    distributed_state: PartialState
        Distributed state from accelerate library. It is used to parallelise over multiple GPUs.
    encoded_text : List[transformers.tokenization_utils_base.BatchEncoding]
        Prompts encoded with a tokeniser, e.g. tokeniser(list_of_strings, padding="max_length", return_tensors="pt").
    target_words: List[str]
        The target words, by default ['1', '2'].
    dtype: torch.dtype, optional
        The dtype of the target token probabilities, by default torch.float16.

    Returns
    -------
    summary_predictions: List[List[float]]
        All the normalised predictions appended to a list of lists. While lists are mutable, making this list a required
        parameter ensures it exists outside this function, so that it can be appended to for all the batched prompts.
    """
    target_words = target_words if target_words is not None else ["1", "2"]
    # summary_predictions = summary_predictions if summary_predictions is not None else [[] for _ in target_words]

    # Get model predictions for the next token
    target_token_ids = tokeniser.encode(target_words, add_special_tokens=False)
    target_token_ids_tensor = torch.tensor(
        target_token_ids
    )  # Convert to tensor for batch processing

    predictions_target_1 = []
    predictions_target_2 = []
    with distributed_state.split_between_processes(
        encoded_text, apply_padding=True
    ) as batched_data:
        for single_batch in tqdm(batched_data):
            # Move the batch to the device
            single_batch = single_batch.to(distributed_state.device)
            prediction_params = (
                single_batch
                if 'causal' in original_model.config.architectures[0].lower()
                else dict(
                    **single_batch,
                    decoder_input_ids=original_model._shift_right(
                        single_batch.input_ids
                    ),
                )
            )
            with torch.no_grad():
                outputs = original_model(**prediction_params)
                predictions = outputs.logits

            log_probabs = torch.log_softmax(
                predictions[:, -1, :], dim=-1, dtype=dtype
            )  # Compute log softmax in one go

            # Extract the log probabilities for the target token IDs
            log_probabs_target = log_probabs[:, target_token_ids_tensor]

            # Convert log probabilities to probabilities
            softmaxed_log_probabs = torch.softmax(
                log_probabs_target, dim=-1, dtype=dtype
            )
            softmaxed_log_probabs = torch.transpose(
                softmaxed_log_probabs, 0, 1
            ).tolist()

            predictions_target_1.extend(softmaxed_log_probabs[0])
            predictions_target_2.extend(softmaxed_log_probabs[1])

    collected_predictions_1 = gather_object(predictions_target_1)
    collected_predictions_2 = gather_object(predictions_target_2)
    # distributed_state.print(
    #     f'length of encoded_texts: {len(encoded_text)}, collected_predictions_1:'
    #     f' {len(collected_predictions_1)} collected_predictions_2: {len(collected_predictions_2)}'
    # )

    return [collected_predictions_1, collected_predictions_2]


def get_mean_of_probabilities(
    predictions_dict: Dict[int, List[List[float]]], target_words
):
    pm_decision_per_target_word = {}
    for j, target_word in enumerate(target_words):
        unbiased_predictions = torch.tensor(
            [predictions[j] for predictions in predictions_dict.values()]
        )
        # print(f'unbiased_predictions: {unbiased_predictions}')
        pm_decision_per_target_word[target_word] = torch.mean(
            unbiased_predictions, dim=0
        )
        # print(f'pm_decision_per_target_word[{target_word}]: {pm_decision_per_target_word[target_word]}')

    # would need to implement this dynamically if more labels were needed.
    # Here, if last word is bigger than 50%, then 1 else 0.
    # print(pm_decision_per_target_word)
    ai_choice = torch.where(pm_decision_per_target_word[target_words[-1]] >= 0.5, 1, 0)

    return ai_choice.tolist()


def dataset_dict_to_langchain_batch_consumable(
    data: DatasetDict, requested_cols: List[str], data_split: str = None
) -> List[dict]:
    """This function takes a DatasetDict and converts into a list of dictionaries that langchain.batch() can consume.

    Parameters
    ----------
    data : DatasetDict
        The dataset to be converted into a list of dictionaries.
    requested_cols : List[str]
        The list of columns that will be used in the prompt template.
    data_split : str, optional
        The data split of the dataset to be converted into a list of dictionaries. If data_split is not provided, it is
        assumed that only the relevant split of data is provided.

    Returns
    -------
    data_for_langchain: List[dict]
        List of dictionaries that a langchain.batch() method can consume.
    """
    requested_data = data[data_split] if data_split is not None else data
    data_for_langchain = []
    for values in zip(*[requested_data[col] for col in requested_cols]):
        row_value = {col: values[index] for index, col in enumerate(requested_cols)}
        data_for_langchain.append(row_value)

    return data_for_langchain


def cot_prompt_decoder(tokeniser, model_outputs):
    rational_split = 'Rational:'
    decoded_completions = [
        # tokeniser specific changes
        ' '.join(prompt.split(rational_split)[:-1]).replace(tokeniser.pad_token, '')
        + f" {prompt.split(rational_split)[-1].replace(EOS_TOKEN, '')}"
        + ENDING_LEE_ET_AL
        for prompt in tokeniser.batch_decode(model_outputs, skip_special_tokens=False)
    ]
    return decoded_completions


def reflexion_prompt_decoder(tokeniser, model_outputs):
    """This function decodes the model output and omits the Reflexion instructions.
    So that, the original prompts is only appended with the Observations.
    E.g. with 2 retries the following would result:
    - Preamble
    - Article
    - Candidate summaries
    - CoT
    - Preferred Summary = <wrong answer>
    - Observation: (this is the reflexion completion from the model).
    - CoT
    - Preferred Summary = <wrong answer>
    - Observation: (this is the reflexion completion from the model).
    - CoT
    - Preferred Summary = <right answer>
    """
    # Since the reflexion instructions are always removed, this ensures consistency when multiple retries are attempted.
    split_string = (
        f'{BOS_USER_TOKEN} You were unsuccessful in rating'
        # f"""{EOS_TOKEN}{SUMMARISATION_REFLEXION_PROMPT}"""
    )
    decoded_completions = tokeniser.batch_decode(
        model_outputs, skip_special_tokens=False
    )
    print('===================\n', decoded_completions)
    decoded_completions = [
        f"""{prompt.split(split_string)[0].replace(tokeniser.pad_token, '')}
Observation: {prompt.split(split_string)[1].split('Observation:')[1].replace(tokeniser.pad_token, '').replace(EOS_TOKEN, '')}"""
        # tokeniser specific changes
        for prompt in decoded_completions
    ]
    return decoded_completions


class DecoderSelector(Enum):
    CoT = cot_prompt_decoder
    Reflexion = reflexion_prompt_decoder


def generate_tokens_with_gpus(
    labeller_model,
    tokeniser,
    distributed_state,
    chain,
    langchain_batch_consumable_prompts: List[Dict],
    batch_size: int,
    max_new_tokens: int,
    pad_to_multiple_of: int = 8,
    decoding_choice: DecoderSelector = DecoderSelector.CoT,
) -> List[str]:
    prompts_to_complete = [
        prompt.text for prompt in chain.batch(langchain_batch_consumable_prompts)
    ]

    batched_dataset = [
        prompts_to_complete[i : i + batch_size]
        for i in range(0, len(prompts_to_complete), batch_size)
    ]
    encoded_batches = [
        tokeniser(
            formatted_prompt,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        for formatted_prompt in batched_dataset
    ]
    completions_per_process = []
    generation_config = (
        GenerationConfig(
            max_new_tokens=max_new_tokens, pad_token_id=tokeniser.pad_token_id
        )
        if 'causal' in labeller_model.config.architectures[0].lower()
        else GenerationConfig(max_new_tokens=max_new_tokens)
    )
    with distributed_state.split_between_processes(
        encoded_batches, apply_padding=True
    ) as batched_data:
        for single_batch in tqdm(batched_data):
            # Move the batch to the device
            single_batch = single_batch.to(distributed_state.device)

            labeller_outputs = labeller_model.generate(
                **single_batch, generation_config=generation_config,
            )
            # labeller_outputs: List[str] that is of batch_size length
            # if reflexion, cut reflexion instructions and append reflexion
            # if insights, add to VDB
            decoded_completions = decoding_choice(
                tokeniser=tokeniser, model_outputs=labeller_outputs
            )
            completions_per_process.extend(decoded_completions)

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    completions_gather = gather_object(completions_per_process)

    # Drop duplicates produced by apply_padding in split_between_processes
    model_completions = completions_gather[: len(langchain_batch_consumable_prompts)]
    return model_completions


def generate_cot_for_prompts_with_gpus(
    dataset,
    labeller_model,
    tokeniser,
    distributed_state,
    reverse: bool = False,
    batch_size: int = 2,
    max_new_tokens: int = 256,
) -> List[str]:
    """Function to assemble and execute CoT prompts utilising langchain.

    References
    ----------
    https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
    """
    # batch_size = batch_size * 2
    template = OPENAI_PREAMBLE + TASK_WITH_COT_LEE_ET_AL
    prompt_template = PromptTemplate.from_template(template)

    # This chain only assembles the prompts
    rationale_prompt_chain = {
        "text": itemgetter(POST_COL),
        "summary1": (
            itemgetter(f"{CANDIDATE_COL}_2")
            if reverse is True
            else itemgetter(f"{CANDIDATE_COL}_1")
        ),
        "summary2": (
            itemgetter(f"{CANDIDATE_COL}_1")
            if reverse is True
            else itemgetter(f"{CANDIDATE_COL}_2")
        ),
    } | prompt_template
    requested_cols = [POST_COL, f"{CANDIDATE_COL}_1", f"{CANDIDATE_COL}_2"]
    list_of_dict_dataset = dataset_dict_to_langchain_batch_consumable(
        data=dataset, requested_cols=requested_cols, data_split='train'
    )

    rationale_completions = generate_tokens_with_gpus(
        labeller_model=labeller_model,
        tokeniser=tokeniser,
        distributed_state=distributed_state,
        chain=rationale_prompt_chain,
        langchain_batch_consumable_prompts=list_of_dict_dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    return rationale_completions


def generate_reflexion_and_cot_completions_with_gpus(
    dataset,
    labeller_model,
    tokeniser,
    distributed_state,
    prompt_col: str,
    batch_size: int = 2,
    max_new_tokens: int = 256,
) -> List[str]:
    """Function to assemble and execute reflexion prompts utilising langchain.

    References
    ----------
    https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
    """
    # generating tokens is cheaper than generating token probabilities, so this is attempting to maximise GPU capacity.
    batch_size = batch_size * 2
    template = (
        "{cot_prompt}{predicted_summary}"
        + f"""{EOS_TOKEN}{SUMMARISATION_REFLEXION_PROMPT.replace('''
''', ' ')}"""
    )
    prompt_template = PromptTemplate.from_template(template)

    # This chain only assembles the prompts
    reflexion_chain = {
        "cot_prompt": itemgetter(prompt_col),
        "predicted_summary": itemgetter("ai_choice"),
    } | prompt_template

    requested_cols = [prompt_col, "ai_choice"]
    list_of_dict_dataset = dataset_dict_to_langchain_batch_consumable(
        data=dataset, requested_cols=requested_cols
    )

    completions = generate_tokens_with_gpus(
        labeller_model=labeller_model,
        tokeniser=tokeniser,
        distributed_state=distributed_state,
        chain=reflexion_chain,
        decoding_choice=DecoderSelector.Reflexion,
        langchain_batch_consumable_prompts=list_of_dict_dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    cot_template = "{previous_attempt}\nRationale:"
    cot_prompt_template = PromptTemplate.from_template(cot_template)
    cot_chain = {
        "previous_attempt": itemgetter('previous_attempt_with_reflexion'),
    } | cot_prompt_template
    cot_with_reflexion_list_of_dict = [
        {'previous_attempt_with_reflexion': completion} for completion in completions
    ]

    cot_completions = generate_tokens_with_gpus(
        labeller_model=labeller_model,
        tokeniser=tokeniser,
        distributed_state=distributed_state,
        chain=cot_chain,
        decoding_choice=DecoderSelector.CoT,
        langchain_batch_consumable_prompts=cot_with_reflexion_list_of_dict,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    return cot_completions


def generate_next_token_probabilities_gpus(
    model,
    tokeniser,
    decoded_reasoning,
    distributed_state,  # object from accelerate
    batch_size: int,
    target_words: List[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[List[float]]:
    """
    References
    ----------
    https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
    """
    target_words = target_words if target_words is not None else ["1", "2"]

    batched_dataset = [
        decoded_reasoning[i : i + batch_size]
        for i in range(0, len(decoded_reasoning), batch_size)
    ]
    encoded_batches = [
        tokeniser(
            formatted_prompt, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )
        for formatted_prompt in batched_dataset
    ]

    # Re-order the split-between processes here!
    distributed_state.print('Probabilities being calculated.')
    probabilities = calculate_target_token_probabilities_with_gpus(
        original_model=model,
        tokeniser=tokeniser,
        encoded_text=encoded_batches,
        target_words=target_words,
        distributed_state=distributed_state,
        dtype=dtype,
    )

    collected_distinct_predictions_1 = probabilities[0][: len(decoded_reasoning)]
    collected_distinct_predictions_2 = probabilities[1][: len(decoded_reasoning)]
    distributed_state.print(
        f'length of encoded_texts: {len(decoded_reasoning)}, collected_distinct_predictions_1: '
        f'{len(collected_distinct_predictions_1)} collected_distinct_predictions_2: '
        f'{len(collected_distinct_predictions_2)}.'
    )
    return [collected_distinct_predictions_1, collected_distinct_predictions_2]


def get_example_str_from_retrieved_doc(retrieved_documents: List[Document]) -> str:
    # {
    #     "type": "example",
    #     "summary1": values[1],
    #     "summary2": values[2],
    #     # The split_string only exists for the model generated reasoning so it makes it easy to split on.
    #     "reasoning": values[3].split(RATIONALES_SPLIT_STRING)[1],
    #     "predicted_label": values[4],
    #     "end_string": end_of_example_string,
    # }
    relevant_content = (
        [
            RETRIEVED_EXAMPLE_TEMPLATE.format(
                text=doc.page_content,
                summary1=doc.metadata.get('summary1'),
                summary2=doc.metadata.get('summary2'),
                chain_of_thought=doc.metadata.get("reasoning"),
                ai_choice=doc.metadata.get('predicted_label'),
            )
            + f" {doc.metadata.get('end_string')}"
            for doc in retrieved_documents
        ]
        if len(retrieved_documents) >= 1
        else []
    )
    return '\n'.join(relevant_content)


class InsightActions(Enum):
    edit = 'edit'
    add = 'add'
    remove = 'remove'
    agree = 'agree'


# def parse_insights_actions(completion: str) -> List[List[str]]:
#     seperator = 'Do at most 4 operations and each existing rule can only get a maximum of 1 operation.'
#     insight_actions = completion.split(seperator)[0].split('\n')
#     insight_actions = [action.strip().split(' ') for action in insight_actions]
#     correctly_parsed_insight_actions = [
#         action for action in insight_actions if len(action) == 3
#     ]
#     return correctly_parsed_insight_actions


def parse_insights_actions(text):
    pattern = (
        r'(AGREE|REMOVE|EDIT|ADD) (\d+): (.+?)(?=(?:AGREE|REMOVE|EDIT|ADD) \d+: |$)'
    )
    matches = re.findall(pattern, text)
    results = [
        (operation, int(number), string.strip())
        for operation, number, string in matches
    ]
    return results


def parse_insights_to_dict(insights: str) -> Dict[str, str]:
    if len(insights) >= 1:
        insights_list = insights.split('\n')
        insights_dict = {
            f'{index}': rule.split(':')[1].strip()
            for index, rule in enumerate(insights_list)
        }
        return insights_dict
    return dict()


def update_insights(insight_actions: List[str], insights: str) -> str:
    correctly_parsed_insight_actions = parse_insights_actions(
        completion=insight_actions[0]
    )
    print(f'correctly_parsed_insight_actions: {correctly_parsed_insight_actions}')
    insights_dict = parse_insights_to_dict(insights)
    # ADD <NEW RULE NUMBER>: <NEW RULE>
    for action, rule_number, rule in correctly_parsed_insight_actions:
        rule_number = rule_number.split(':')[0].strip()
        if action == InsightActions.edit or action == InsightActions.add:
            insights_dict[rule_number] = rule
        elif action == InsightActions.remove:
            insights_dict.pop(rule_number, None)

    insights_str = ''
    for index, rule in enumerate(list(insights_dict.values())):
        insights_str += f'{index}: {rule}\n'
    return insights_str


def generate_insights_successful(
    dataset,
    labeller_model,
    tokeniser,
    distributed_state,
    insights: str,
    reverse: bool = False,
    max_new_tokens: int = 2048,
) -> str:
    """This function is looping over the insight generations because it is simpler to make edit, update and add
    operations (on the same object) sequentially rather than with parallelised and batched operations...

    Note
    ----
    At the moment, this function creates 1 prompt out of every 5 successful exercises to compare.
    A good summarisation labeling task is ~70% accurate.
    So this method O(7N/50) slow."""
    # generating tokens is cheaper than generating token probabilities, so this is attempting to maximise GPU capacity.
    prompt_col = (
        'reversed_prompt_used_to_predict'
        if reverse is True
        else 'ordered_prompt_used_to_predict'
    )
    prompt_template = PromptTemplate.from_template(ALL_SUCCESSES_INSIGHTS_TEMPLATE)

    list_of_dict_dataset = dataset_dict_to_langchain_batch_consumable(
        data=dataset, requested_cols=[prompt_col]
    )
    # Get only thought, action, observation-s for successful trials, then group them 5 at a time.
    successes = [
        element.get(prompt_col).split(RATIONALES_SPLIT_STRING)[1]
        for element in list_of_dict_dataset
    ]
    list_of_dict_dataset = [
        {"successes": "\n\n".join(successes[i : i + 5])}
        for i in range(0, len(successes), 5)
    ]
    successful_insights_chain = {
        "success_history": itemgetter("successes"),
        "existing_rules": itemgetter("insights"),
    } | prompt_template

    for prompt in list_of_dict_dataset:
        prompt['insights'] = insights
        template = successful_insights_chain.invoke(prompt)
        # distributed_state.print(f'successful insights template: {template}, {type(template)}')
        # distributed_state.print(f'successful insights template: {template.text}, prompt: {prompt}, {type(prompt)}')
        # prompts_to_complete = [
        #     prompt.text for prompt in template
        # ]

        tokenised_prompts = tokeniser(
            template.text, padding=True, return_tensors="pt"
        ).to(torch.device(distributed_state.device))

        with torch.no_grad():
            labeller_outputs = labeller_model.generate(
                **tokenised_prompts,
                generation_config=(
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokeniser.pad_token_id,
                    )
                    if 'causal' in labeller_model.config.architectures[0].lower()
                    else GenerationConfig(max_new_tokens=max_new_tokens)
                ),
            )
        decoded_insight_actions = tokeniser.batch_decode(
            labeller_outputs, skip_special_tokens=True
        )
        distributed_state.print(
            f'successful insight_actions: {decoded_insight_actions}'
        )
        insights = update_insights(
            insight_actions=decoded_insight_actions, insights=insights
        )

    return insights


def generate_insights_with_comparisons(
    dataset,
    labeller_model,
    tokeniser,
    distributed_state,
    insights: str,
    reverse: bool = False,
    max_new_tokens: int = 2048,
) -> str:
    """This function is looping over the insight generations because it is simpler to make edit, update and add
    operations (on the same object) sequentially rather than with parallelised and batched operations...

    Note
    ----
    At the moment, this function creates 1 prompt out of every successfully retried exercise.
    Will need to quantify how often this occurs.
    But it should be a subset of the 30% of exercises."""

    # Get only thought, action, observation-s for successful trials, then group them 5 at a time.
    prompt_col = (
        'reversed_prompt_used_to_predict'
        if reverse is True
        else 'ordered_prompt_used_to_predict'
    )
    requested_cols = [POST_COL, f"{CANDIDATE_COL}_1", f"{CANDIDATE_COL}_2", prompt_col]
    list_of_dict_dataset = dataset_dict_to_langchain_batch_consumable(
        data=dataset, requested_cols=requested_cols
    )
    for element in list_of_dict_dataset:
        rationale_action_observation = element.get(prompt_col).split(
            RATIONALES_SPLIT_STRING
        )[1]
        element['success_trajectory'] = rationale_action_observation
        # Data only makes it into here if it succeeded after failure. So take everything until latest retry to get
        # failed trajectory.
        element['fail_trajectory'] = 'Observation'.join(
            rationale_action_observation.split('Observation')[:-1]
        )

    # generating tokens is cheaper than generating token probabilities, so this is attempting to maximise GPU capacity.
    prompt_template = PromptTemplate.from_template(
        FAIL_SUCCESS_COMPARISON_INSIGHTS_TEMPLATE
    )
    task_template = OPENAI_PREAMBLE + TASK_WITH_COT_LEE_ET_AL.split("Rationale:")[0]
    task_prompt_template = PromptTemplate.from_template(task_template)
    rationale_prompt_chain = (
        {
            "text": itemgetter(POST_COL),
            "summary1": itemgetter(f"{CANDIDATE_COL}_2"),
            "summary2": itemgetter(f"{CANDIDATE_COL}_1"),
        }
        | task_prompt_template
        if reverse is True
        else {
            "text": itemgetter(POST_COL),
            "summary1": itemgetter(f"{CANDIDATE_COL}_1"),
            "summary2": itemgetter(f"{CANDIDATE_COL}_2"),
        }
        | task_prompt_template
    )
    comparison_insights_chain = {
        "task": rationale_prompt_chain,
        "success_history": itemgetter("success_trajectory"),
        "fail_history": itemgetter("fail_trajectory"),
        "existing_rules": itemgetter("insights"),
    } | prompt_template

    for prompt in list_of_dict_dataset:
        prompt['insights'] = insights
        prompts_to_complete = comparison_insights_chain.invoke(prompt)

        tokenised_prompts = tokeniser(
            prompts_to_complete.text, padding=True, return_tensors="pt"
        ).to(torch.device(distributed_state.device))

        with torch.no_grad():
            labeller_outputs = labeller_model.generate(
                **tokenised_prompts,
                generation_config=(
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokeniser.pad_token_id,
                    )
                    if 'causal' in labeller_model.config.architectures[0].lower()
                    else GenerationConfig(max_new_tokens=max_new_tokens)
                ),
            )
        decoded_insight_actions = tokeniser.batch_decode(
            labeller_outputs, skip_special_tokens=True
        )
        distributed_state.print(
            f'comparison insight_actions: {decoded_insight_actions}'
        )
        insights = update_insights(
            insight_actions=decoded_insight_actions, insights=insights
        )

    return insights


def get_documents_from_data(
    dataset: DatasetDict, negative_examples: bool = False, reverse: bool = False
) -> List[Document]:
    prompt_col = (
        'reversed_prompt_used_to_predict'
        if reverse is True
        else 'ordered_prompt_used_to_predict'
    )
    columns_needed = [
        POST_COL,
        f"{CANDIDATE_COL}_1",
        f"{CANDIDATE_COL}_2",
        prompt_col,
        "ai_choice",
    ]
    end_of_example_string = (
        "END OF BAD EXAMPLE." if negative_examples is True else "END OF GOOD EXAMPLE."
    )
    # Get only thought, action, observation-s for successful trials, then group them 5 at a time.
    documents = [
        Document(
            page_content=values[0],
            metadata={
                "type": "example",
                "summary1": values[1],
                "summary2": values[2],
                # The split_string only exists for the model generated reasoning so it makes it easy to split on.
                "reasoning": values[3].split(RATIONALES_SPLIT_STRING)[1],
                "predicted_label": values[4],
                "end_string": end_of_example_string,
            },
        )
        for values in zip(*[dataset[col] for col in columns_needed])
    ]
    return documents


def generate_cot_with_insights_and_examples_prompts_with_gpus(
    dataset,
    labeller_model,
    tokeniser,
    distributed_state,
    reverse: bool = False,
    batch_size: int = 2,
    max_new_tokens: int = 256,
    insights: str = None,
    vdb_retriever: VectorStoreRetriever = None,
) -> List[str]:
    """Function to assemble and execute CoT with dynamic examples and or insights prompts utilising langchain.

    References
    ----------
    https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/phi2.py
    """
    batch_size = batch_size * 2
    preamble = OPENAI_PREAMBLE
    if insights is not None:
        preamble += INSIGHTS
    if vdb_retriever is not None:
        preamble += COT_EXAMPLE

    template = preamble + TASK_WITH_COT_LEE_ET_AL
    prompt_template = PromptTemplate.from_template(template)

    chain_dict = {
        "text": itemgetter(POST_COL),
        "summary1": (
            itemgetter(f"{CANDIDATE_COL}_2")
            if reverse is True
            else itemgetter(f"{CANDIDATE_COL}_1")
        ),
        "summary2": (
            itemgetter(f"{CANDIDATE_COL}_1")
            if reverse is True
            else itemgetter(f"{CANDIDATE_COL}_2")
        ),
    }
    if insights is not None:
        chain_dict.update({"insights": insights})
    if vdb_retriever is not None:
        chain_dict.update(
            {
                "example": itemgetter(POST_COL)
                | vdb_retriever
                | RunnableLambda(get_example_str_from_retrieved_doc)
            }
        )

    # This chain only assembles the prompts
    cot_with_insights_and_examples_chain = chain_dict | prompt_template
    requested_cols = [POST_COL, f"{CANDIDATE_COL}_1", f"{CANDIDATE_COL}_2"]
    list_of_dict_dataset = dataset_dict_to_langchain_batch_consumable(
        data=dataset,
        requested_cols=requested_cols,  # data_split='train', called with subset of data.
    )
    rationale_completions = generate_tokens_with_gpus(
        labeller_model=labeller_model,
        tokeniser=tokeniser,
        distributed_state=distributed_state,
        chain=cot_with_insights_and_examples_chain,
        langchain_batch_consumable_prompts=list_of_dict_dataset,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    return rationale_completions
