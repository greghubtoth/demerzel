import json

import datasets
import pandas as pd
import requests
import torch
from datasets import concatenate_datasets
from peft import TaskType

from nyx.constants import (CANDIDATE_COL, CHOICE_COL, COMPARISON_DATA_PATH,
                           INFO_COL, POST_COL, REDDIT_DATA_URL,
                           SFT_DATA_OUTPUT_PATH, SUBREDDIT_COL, SUMMARY_COL,
                           TEXT_COL)


def precision_enumerator(requested_precision: str) -> torch.dtype or str:
    """Enumerator to overcome torch precision serialisation issue.
    Google whether this can be replaced by a torch enum.
    """
    if requested_precision == 'float32':
        return torch.float32
    elif requested_precision == 'float16':
        return torch.float16
    elif requested_precision == 'bfloat16':
        return torch.bfloat16
    elif requested_precision == 'int8':
        return torch.int8
    elif requested_precision == 'auto':
        return 'auto'
    else:
        raise NotImplementedError(
            'Precision enumerator only supports float32, float16, bfloat16 and int8.'
        )


def download_jsonl(url):
    """Utility function to download jsonl files from a website.

    Notes
    -----
    This function is employed in download_and_save_reddit_data().
    """
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        data = []
        for line in response.iter_lines():
            # Convert each line into a Python dictionary
            json_line = json.loads(line.decode("utf-8"))
            data.append(json_line)
        return data
    else:
        raise Exception(
            "Failed to download file, status code: " + str(response.status_code)
        )


DATA_REF_MAPPING = {"valid": "validation"}


def download_and_save_reddit_data(url: str = REDDIT_DATA_URL):
    """Utility function to download jsonl files from a website.

    Notes
    -----
    Utility function to obtain reddit data from OpenAI website which can be employed for behaviour cloning and RL, i.e.
    fine-tuning (FT).
    """
    conversion_dict = {}
    for data_split in ["train", "test", "valid"]:
        dataset_reference_name = DATA_REF_MAPPING.get(data_split, data_split)
        url_path = url.format(split=data_split)
        jsonl_data = download_jsonl(url_path)
        pd_df = pd.json_normalize(jsonl_data)
        print(data_split, pd_df.shape)
        conversion_dict[dataset_reference_name] = datasets.Dataset.from_pandas(pd_df)

    datasets.DatasetDict(conversion_dict).save_to_disk(SFT_DATA_OUTPUT_PATH)
    print(f"Successfully saved data to:\n{SFT_DATA_OUTPUT_PATH}")


def round_dictionary_values(some_dict: dict, n_decimals: int = 2) -> dict:
    """Function to round values in the dictionary.

    Parameters
    ----------
    some_dict : dict
        Dictionary to round.
    n_decimals : int
        Number of decimals to round to, by default 2.

    Returns
    -------
    some_dict: dict
    """
    some_dict = {k: round(v, n_decimals) for k, v in some_dict.items()}
    return some_dict


def print_number_of_trainable_model_parameters(model) -> str:
    """Function to print the number of trainable parameters of a model.

    References
    ----------
    deeplearning.ai course
    https://www.deeplearning.ai/courses/generative-ai-with-llms/
    """
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}"
        f"\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )


def restructure_and_save_data(
    comparison_dataset: datasets.DatasetDict,
    local_path: str = COMPARISON_DATA_PATH,
    choice_col: str = CHOICE_COL,
    info_col: str = INFO_COL,
    post_col: str = POST_COL,
    subreddit_col: str = SUBREDDIT_COL,
    summary_col: str = SUMMARY_COL,
    text_col: str = TEXT_COL,
    candidate_col: str = CANDIDATE_COL,
) -> str:
    """Restructuring and saving OpenAI labelled reddit comparison dataset.

    Notes
    -----
    This is the data for Reward Modelling.
    """
    conversion_dataset_dict = {}
    missing_dps_dropped = 0
    for split in comparison_dataset:
        print(split)
        split_list = []
        for example in comparison_dataset[split]:
            choice = example[choice_col]
            post = example[info_col][post_col]
            subreddit = example[info_col][subreddit_col]
            cand_summary_1 = example[summary_col][0][text_col]
            cand_summary_2 = example[summary_col][1][text_col]

            # Filtering for valid rows. Some missing points found in validation set.
            # {'train': 0, 'validation': 2284}
            if all(
                dp is not None
                for dp in [choice, post, subreddit, cand_summary_1, cand_summary_2]
            ):
                split_list.append(
                    {
                        subreddit_col: subreddit,
                        post_col: post,
                        choice_col: choice,
                        f"{candidate_col}_1": cand_summary_1,
                        f"{candidate_col}_2": cand_summary_2,
                    }
                )
            else:
                missing_dps_dropped += 1
        conversion_dataset_dict[split] = datasets.Dataset.from_list(split_list)

    dataset_dict = datasets.DatasetDict(conversion_dataset_dict)
    dataset_dict.save_to_disk(local_path)
    print(f"Successfully saved data to:\n{local_path}.")
    print(f"{missing_dps_dropped} datapoints have been dropped due to missing values.")
    return local_path


def concat_to_make_unbiased_data(
    dataset_a: datasets.Dataset, dataset_b: datasets.Dataset
) -> datasets.DatasetDict:
    # this function combines two datasets dicts of along rows for each split. I.e. they have to have the sampe splits
    # and features.
    conversion_dataset_dict = {}
    for split in dataset_a:
        concatted_data = concatenate_datasets([dataset_a[split], dataset_b[split]])
        conversion_dataset_dict[split] = concatted_data

    return datasets.DatasetDict(conversion_dataset_dict)


def get_task_type(model):
    task_type = (
        TaskType.CAUSAL_LM
        if 'causal' in model.config.architectures[0].lower()
        else TaskType.SEQ_2_SEQ_LM
    )
    return task_type
