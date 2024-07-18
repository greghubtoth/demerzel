#!/usr/bin/env python
# coding: utf-8
import json
import os
import time
import uuid

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, Trainer, TrainingArguments)
from nyx.evaluation import quantitative_comparison

# In[1]:


# !pip install peft -U


# # Supervised Fine Tuning

# In[24]:


tqdm.pandas()

from pathlib import Path

import pandas as pd

TRAIN_BATCH_SIZE = 4
EVALUATION_BATCH_SIZE = 4
LEARNING_RATE = 1.41e-5  # 1e-3
LORA_PARAM_R = 8
LORA_PARAM_ALPHA = 16
LORA_PARAM_TARGET_MODULES = {
    "bigscience/mt0-small": ["q", "v"],
    "microsoft/phi-1_5": ["q_proj", "v_proj"],
    "microsoft/Phi-3-mini-4k-instruct": ["qkv_proj"],
    "microsoft/Phi-3-medium-4k-instruct": ["qkv_proj"],
}
# PRECISION = torch.float32
PRECISION_NAME = 'float16'
DEVICE = "cuda"  # 0 if torch.cuda.is_available() else "cpu"
CHOSEN_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # "bigscience/mt0-small" #"google/flan-t5-large"
TESTING = True
RUN_ID = uuid.uuid4().hex
print(RUN_ID)


print(f"Model: {CHOSEN_MODEL} will be trained on device: {DEVICE}.")


# ### Developed utility functions
# - For details see [Bath github link](https://github.bath.ac.uk/gt566/ai-msc-dissertation/blob/dissertation-experienced-ft/nyx/dissertation/utils.py)

# In[25]:


from nyx.constants import (COMMON_OUTPUT_PATHS, METRICS_PATH,
                           SFT_DATA_OUTPUT_PATH, SFT_OUTPUT_DIR,
                           SFT_PEFT_ADAPTER_PATH, SFT_PEFT_MERGED_MODEL_PATH)
from nyx.utils import (download_and_save_reddit_data, get_task_type,
                       precision_enumerator,
                       print_number_of_trainable_model_parameters,
                       round_dictionary_values)

COMMON_OUTPUT_PATHS = COMMON_OUTPUT_PATHS.format(RUN_ID=RUN_ID)
METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS)
COMMON_OUTPUT_PATHS = COMMON_OUTPUT_PATHS.format(RUN_ID=RUN_ID)
SFT_OUTPUT_DIR = SFT_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS)
SFT_PEFT_ADAPTER_PATH = SFT_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS
)
SFT_PEFT_MERGED_MODEL_PATH = SFT_PEFT_MERGED_MODEL_PATH.format(
    COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS
)


PRECISION = precision_enumerator(PRECISION_NAME)
PRECISION


# ## Load model and data

# In[26]:


try:
    original_model = AutoModelForSeq2SeqLM.from_pretrained(
        CHOSEN_MODEL, torch_dtype=PRECISION, attn_implementation="flash_attention_2"
    )
except ValueError:
    original_model = AutoModelForCausalLM.from_pretrained(
        CHOSEN_MODEL, torch_dtype=PRECISION, attn_implementation="flash_attention_2"
    )

tokenizer = AutoTokenizer.from_pretrained(
    CHOSEN_MODEL, padding_side="left"
)  # model_max_length=512


# original_model.to(torch.device(DEVICE))
# print("Push models and data to GPU for efficiency.")

print(print_number_of_trainable_model_parameters(original_model))


# In[27]:


filtered_reddit_summarisation_data = Path(SFT_DATA_OUTPUT_PATH)
if not filtered_reddit_summarisation_data.is_dir():
    print("Downloading and saving filtered reddit data.")
    download_and_save_reddit_data()

dataset = load_from_disk(SFT_DATA_OUTPUT_PATH)
dataset


# In[28]:


if TESTING is True:
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(100))
    dataset["validation"] = dataset["validation"].select(range(50))
    # dataset = dataset.filter(
    #     lambda example, index: index % 4680 == 0, with_indices=True
    # )
dataset


# In[38]:


tokenizer.pad_token = (
    tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
)


def tokenize_function(example):
    start_prompt = "Summarize the following reddit post.\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + post + end_prompt for post in example["post"]]
    example['check'] = prompt
    example["input_ids"] = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        return_tensors="pt",  # padding=True
    ).input_ids.to(torch.device(DEVICE))
    example["labels"] = tokenizer(
        example["summary"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",  # padding=True
    ).input_ids.to(torch.device(DEVICE))

    return example


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    [
        "id",
        "subreddit",
        "post",
        "summary",
    ]
)


# In[39]:


print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)


# ## Train PEFT adapter

# In[31]:


# Checking for layers to apply LoRA. Selecting the query and value layers are the most
# basic implementation according to the paper. They are refered to as q and v here.
# print(original_model)


# In[34]:


lora_config = LoraConfig(
    # Determines the size of LoRA matrices. x*r * r*y = x*y
    r=LORA_PARAM_R,
    # scaling coefficient. Paper mentions it is important because the adjustments are small compared
    # to the rest of the model.
    lora_alpha=LORA_PARAM_ALPHA,
    # Variable target_modules determines what layers are fine-tuned, see architecture above.
    # Simplest case scenario based on the original paper.
    target_modules=LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL],
    lora_dropout=0.05,
    bias="none",
    task_type=get_task_type(model=original_model),
)


# In[35]:


peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))


# In[40]:


# common_folder_path = f"./models/openai-subreddit-data/{CHOSEN_MODEL}/supervised-fine-tuning"
# output_dir = (
#     f"{common_folder_path}/peft-dialogue-summary-training-{str(int(time.time()))}"
# )
# peft_model_path = f"{common_folder_path}/peft-dialogue-summary-checkpoint-local"

peft_training_args = TrainingArguments(
    output_dir=SFT_OUTPUT_DIR,
    # auto_find_batch_size=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,  # Higher learning rate than full fine-tuning.
    # num_train_epochs=3,
    save_steps=5_000,
    logging_steps=1,
    max_steps=len(tokenized_datasets["train"])
    // TRAIN_BATCH_SIZE,  # number of training data * 2, i.e. go over all data-summary pairs twice.
)

peft_trainer = Trainer(
    model=peft_model.to(
        torch.device(DEVICE)
    ),  # Important to train on Mac Chip GPU equivalent
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)


# In[41]:


start = time.time()

peft_trainer.train()
end = time.time()

duration = end - start
print(end)
print(f"Training for 1 epoch took {round(duration, 2)} seconds to execute.")

peft_trainer.model.save_pretrained(SFT_PEFT_ADAPTER_PATH)
tokenizer.save_pretrained(SFT_PEFT_ADAPTER_PATH)


# ## Load PEFT adapter

# In[15]:


# Needed to add adapter_config.json to folder and change name of the model from pytorch_model.bin to adapter_model.bin
# adapter_checkpoint_path = f"/Users/gtoth/PycharmProjects/LLM-jupyter-notebooks/openai-subreddit-data-flan-t5-large/peft-dialogue-summary-checkpoint-local-6k"

# adapter_checkpoint_path = f"{common_folder_path}/peft-dialogue-summary-checkpoint-local"
# trained_model = AutoModelForSeq2SeqLM.from_pretrained(
#     CHOSEN_MODEL, torch_dtype=PRECISION
# )
# try:
#     trained_model = AutoModelForSeq2SeqLM.from_pretrained(
#         CHOSEN_MODEL, torch_dtype=PRECISION
#     )
# except ValueError:
#     trained_model = AutoModelForCausalLM.from_pretrained(
#         CHOSEN_MODEL, torch_dtype=PRECISION
#     )
# print("ok")


# ## _Comparing PEFT and Baseline model generations (with ROUGE)_

# In[18]:


# %timeit quantitative_comparison(peft_model)

# In[20]:


N_EVAL_SAMPLES = int(len(tokenized_datasets['test']) * 0.15)

start = time.time()

baseline_model_generation = quantitative_comparison(
    original_model,
    dataset,
    tokenizer,
    n_samples_to_evaluate=N_EVAL_SAMPLES,
    batch_size=EVALUATION_BATCH_SIZE,
    device=DEVICE,
)
# peft_checkpoint_model = PeftModel.from_pretrained(original_model, SFT_PEFT_ADAPTER_PATH)

peft_config = PeftConfig.from_pretrained(SFT_PEFT_ADAPTER_PATH)
# to initiate with random weights
peft_config.init_lora_weights = False
original_model.add_adapter(peft_config)
original_model.enable_adapters()
original_model.to(torch.device(DEVICE))

peft_checkpoint_generation = quantitative_comparison(
    original_model,  # peft enabled model
    dataset,
    tokenizer,
    n_samples_to_evaluate=N_EVAL_SAMPLES,
    batch_size=EVALUATION_BATCH_SIZE,
    device=DEVICE,
)

end = time.time()

duration = end - start
print(
    f"Evaluating N={N_EVAL_SAMPLES} samples took {round(duration, 2)} seconds to execute."
)

human_baseline_answer = dataset["test"][0:N_EVAL_SAMPLES]["summary"]

zipped_summaries = list(
    zip(human_baseline_answer, peft_checkpoint_generation, baseline_model_generation)
)

df = pd.DataFrame(
    zipped_summaries,
    columns=[
        "human_baseline_answer",
        "peft_checkpoint_generation",
        "baseline_model_generation",
    ],
)
df.head()
print(df.shape)


# In[21]:


rouge = evaluate.load("rouge")

original_model_results = rouge.compute(
    predictions=baseline_model_generation,
    references=human_baseline_answer[0 : len(baseline_model_generation)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_checkpoint_generation,
    references=human_baseline_answer[0 : len(peft_checkpoint_generation)],
    use_aggregator=True,
    use_stemmer=True,
)

original_model_results = round_dictionary_values(original_model_results)
# instruct_model_results = round_dictionary_values(instruct_model_results)
peft_model_results = round_dictionary_values(peft_model_results)
print("ORIGINAL MODEL:")
print(original_model_results)
# print('INSTRUCT MODEL:')
# print(instruct_model_results)
print("PEFT MODEL:")
print(peft_model_results)


# In[27]:


METRICS_PATH


# In[29]:


if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)

data_path = f'{METRICS_PATH}/sft-results.json'

results_dict = {
    'rouge-metric-baseline-model': original_model_results,
    'rouge-metric-sft-model': peft_model_results,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'evaluation_batch_size': EVALUATION_BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'lora_param_r': LORA_PARAM_R,
    'lora_param_alpha': LORA_PARAM_ALPHA,
    'lora_param_target_modules': LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL],
    'precision': PRECISION_NAME,
    'device': DEVICE,
    'chosen_model': CHOSEN_MODEL,
    'testing': TESTING,
    'run_id': RUN_ID,
    'gpu_type': torch.cuda.get_device_name(),


}
with open(data_path, 'w') as file:
    json.dump(results_dict, file)


# In[22]:


print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

improvement = np.array(list(peft_model_results.values())) - np.array(
    list(original_model_results.values())
)
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value * 100:.2f}%')


# ## Merge and save peft model (with base model)
# So that, it can be loaded in as a Reward Moldel.

# In[30]:


model = peft_model.merge_and_unload()
model.save_pretrained(SFT_PEFT_MERGED_MODEL_PATH)


# # END
