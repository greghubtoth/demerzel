#!/usr/bin/env python
# coding: utf-8

# # AI labelled data generation and Reward Modelling

import os
# In[1]:
import sys

# Add the local library path

# Get the current working directory
cwd = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(cwd)

print("Current Working Directory:", cwd)
print("Parent Directory:", parent_dir)
# HEX commands are too restrictive.
sys.path.append(os.path.abspath(parent_dir))
# python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

import json
import random
import uuid

from nyx.constants import COMMON_OUTPUT_PATHS, METRICS_PATH
from nyx.data_generation import Controller
from nyx.data_generation.settings import BASELINE_LEE_ET_AL
from nyx.data_loaders import HumanEvaluatedDataLoader

# In[2]:


RANDOM_SEED = 42
# PRECISION = torch.float32
PRECISION_NAME = 'bfloat16'
DEVICE = "cuda"
LABELLER_MODEL = "bigscience/mt0-small"
# "Qwen/Qwen2-0.5B-Instruct"
# "stabilityai/stablelm-2-zephyr-1_6b"
# "microsoft/phi-1_5"
# "microsoft/Phi-3-mini-4k-instruct"
# "google/flan-t5-large"
# "google/flan-t5-xl"
# "bigscience/mt0-small"
# "bigscience/mt0-large"
#
GEMMA_PATH = "/Users/owner/Documents/AI_MSc/13.Dissertation/experiments/labelling-model/gemma-2b-it/"
RUN_ID = uuid.uuid4().hex  # "test-multi-gpu-set-up"  #
TESTING = True
print(f'Employing model: {LABELLER_MODEL} on device: {DEVICE}.')
print(f'RUN_ID: {RUN_ID}')
# In[ ]:


config = {
    'llm_model_name': LABELLER_MODEL,  # LABELLER_MODEL, # GEMMA_PATH, # 70B param model
    'precision_name': PRECISION_NAME,
    'device': DEVICE,
    # 'dataset': data,
    'batch_size': 2,
    'run_id': RUN_ID,
    'max_new_tokens': 512,
}


data_generator = Controller(
    labelling_method=BASELINE_LEE_ET_AL,  # Tóth et al., (Ablation)
    labelling_config=config,
    data_loader=HumanEvaluatedDataLoader,
)

if TESTING is True:
    # indices = random.sample(range(0, 92859), 100)
    # print(indices)
    data_generator.data_to_label["train"] = data_generator.data_to_label[
        "train"
    ].select(
        range((100))
    )  # .select(indices)
    data_generator.data_to_label["validation"] = data_generator.data_to_label[
        "validation"
    ].select(range(50))


# print(data_generator.data_to_label)
data_generator.label_data()
data_generator.report_on_performance()
# (data_generator.data_to_label['train'])


# In[21]:


# from typing import List
# from datasets import DatasetDict
# from pprint import pprint
# def dataset_dict_to_langchain_batch_consumable(data: DatasetDict,
#                                                requested_cols: List[str],
#                                                data_split: str = 'train', ) -> List[dict]:
#     requested_data = data[data_split]
#     data_for_langchain = []
#     for values in zip(*[requested_data[col] for col in requested_cols]):
#         # print(values)
#         row_value = {col: values[index] for index, col in enumerate(requested_cols)}
#         data_for_langchain.append(row_value)

#     return data_for_langchain
# pprint(dataset_dict_to_langchain_batch_consumable(data_generator.data_to_label, ['post', 'candidate_summary_1', 'candidate_summary_2']))

COMMON_OUTPUT_PATHS = COMMON_OUTPUT_PATHS.format(RUN_ID=RUN_ID)
METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS)

if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)

data_path = f'{METRICS_PATH}/data-generation-info.json'

results_dict = {
    'run_id': RUN_ID,
    'labeller_model': LABELLER_MODEL,
    'precision': PRECISION_NAME,
    'duration': data_generator.labelling_duration,
    'n_gpus_available': data_generator.n_gpus_available,
    'gpu_type': data_generator.gpu_type,
}

with open(data_path, 'w') as file:
    json.dump(results_dict, file)

print('Data generation done and the configuration info is saved.')
print(data_path)
# # END
