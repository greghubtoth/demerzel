#!/usr/bin/env python
# coding: utf-8

# ## Reward Modelling

# In[1]:


#!pip install peft trl weaviate-client rouge_score nltk


# In[2]:


import json
import os
import time
import uuid
from typing import List

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from trl import (AutoModelForCausalLMWithValueHead,
                 AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer,
                 RewardConfig, RewardTrainer, create_reference_model)
from trl.core import LengthSampler

# from unsloth import FastLanguageModel
# from unsloth import is_bfloat16_supported

TESTING = False

PRECISION_NAME = 'float16'
DEVICE = "cuda"
CHOSEN_MODEL = "microsoft/phi-1_5"  # "Qwen/Qwen2-7B-Instruct"
# "microsoft/phi-1_5" #"bigscience/mt0-small" # "google/flan-t5-large" "stabilityai/stablelm-2-zephyr-1_6b"
RANDOM_SEED = 42
RUN_ID = "be140fa23e9e4452a7801060e9e7d6ba"  # SFT MODEL # uuid.uuid4().hex

LORA_PARAM_TARGET_MODULES = {
    "bigscience/mt0-small": ["q", "v"],
    "microsoft/phi-1_5": ["q_proj", "k_proj", "v_proj"],
    "microsoft/Phi-3-mini-4k-instruct": ["qkv_proj"],
    "microsoft/Phi-3-medium-4k-instruct": ["qkv_proj"],
    "unsloth/Qwen2-7B-Instruct-bnb-4bit": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "Qwen/Qwen2-7B-Instruct": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

RM_LORA_PARAM_R = 16
RM_LORA_PARAM_ALPHA = 16
RM_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL] + [
    # "dense",
    # "out_proj",
    "score"
]
RM_TRAIN_BATCH_SIZE = 10
RM_LEARNING_RATE = 5e-5
RM_TRAIN_DATA_RUN_ID = "8cb27c1524fe4f7f817fa524859bab07"  # generate data ID

RL_LORA_PARAM_R = 16
RL_LORA_PARAM_ALPHA = 16
RL_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL]
RL_TRAIN_BATCH_SIZE = 2
RL_TRAIN_MINI_BATCH_SIZE = 1
RL_LEARNING_RATE = 1.41e-7
RL_N_EPOCHS = 1


# In[3]:


import sys
from pathlib import Path

path = Path.cwd().parent.absolute()
nyx_path = f'{path}/'
print(nyx_path)
sys.path.append(nyx_path)


# In[4]:


from nyx.constants import (COMMON_OUTPUT_PATHS, COMPARISON_DATA_PATH,
                           METRICS_PATH, RM_OUTPUT_DIR, RM_PEFT_ADAPTER_PATH,
                           RM_PEFT_MERGED_MODEL_PATH, RM_TRAIN_DATA_PATH,
                           SFT_DATA_OUTPUT_PATH, SFT_PEFT_ADAPTER_PATH,
                           SFT_PEFT_MERGED_MODEL_PATH)
from nyx.data_generation.evaluators import AILabelEvaluator
from nyx.data_generation.prompts.model_specific_tokens import (
    QWEN_BOS_ASSISTANT, QWEN_BOS_USER, QWEN_EOS)
from nyx.evaluation import quantitative_comparison
from nyx.utils import (get_task_type, precision_enumerator,
                       print_number_of_trainable_model_parameters,
                       round_dictionary_values)

common_output_path = COMMON_OUTPUT_PATHS.format(RUN_ID=RUN_ID)

SFT_PEFT_ADAPTER_PATH = SFT_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)
SFT_PEFT_MERGED_MODEL_PATH = SFT_PEFT_MERGED_MODEL_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)

# if generated data from a different run needs to be utilised.
if RM_TRAIN_DATA_RUN_ID is not None:
    rm_common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=RM_TRAIN_DATA_RUN_ID)
    RM_TRAIN_DATA_PATH = RM_TRAIN_DATA_PATH.format(COMMON_OUTPUT_PATHS=rm_common_path)
else:  # utilise current run_id
    RM_TRAIN_DATA_PATH = RM_TRAIN_DATA_PATH.format(
        COMMON_OUTPUT_PATHS=common_output_path
    )

OUTPUT_RUN_ID = uuid.uuid4().hex
print(f'RM results will be saved to run_id directory: {OUTPUT_RUN_ID}')
new_output_path = COMMON_OUTPUT_PATHS.format(RUN_ID=OUTPUT_RUN_ID)

RM_OUTPUT_DIR = RM_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=new_output_path)
RM_PEFT_ADAPTER_PATH = RM_PEFT_ADAPTER_PATH.format(COMMON_OUTPUT_PATHS=new_output_path)
RM_PEFT_MERGED_MODEL_PATH = RM_PEFT_MERGED_MODEL_PATH.format(
    COMMON_OUTPUT_PATHS=new_output_path
)

METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=new_output_path)

PRECISION = precision_enumerator(PRECISION_NAME)
PRECISION


# ### Load model and data

# In[5]:


# Original dataset with train, test and validation
comparison_dataset = load_from_disk(COMPARISON_DATA_PATH)

# Train dataset with generated AI labels
comparison_train_dataset = load_from_disk(RM_TRAIN_DATA_PATH)

if TESTING is True:
    comparison_dataset["train"] = comparison_dataset["train"].select(range(200))
    comparison_train_dataset["train"] = comparison_train_dataset["train"].select(
        range(50)
    )
    comparison_dataset["validation"] = comparison_dataset["validation"].select(
        range(50)
    )
else:
    comparison_dataset = comparison_dataset.filter(
        lambda example, index: index % 10 == 0, with_indices=True
    )
print(comparison_dataset)
print(comparison_train_dataset)


# In[6]:


# References
# ----------
# https://colab.research.google.com/drive/1W0j3rP8WpgxRdUgkb5l6E00EEVyjEZGk?usp=sharing#scrollTo=QmUBVEnvCDJv


max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
# sft_model, tokenizer = FastLanguageModel.from_pretrained( # FastLanguageModel, AutoModelForCausalLM
#     SFT_PEFT_ADAPTER_PATH, #SFT_PEFT_ADAPTER_PATH,  # YOUR MODEL YOU USED FOR TRAINING
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
# )
# sft_model.config.pad_token_id = sft_model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(SFT_PEFT_ADAPTER_PATH, padding_side='left')
tokenizer.pad_token = (
    tokenizer.pad_token
    if tokenizer.pad_token is not None
    else tokenizer.eos_token  # '<|endoftext|>'
)
tokenizer.padding_side = 'left'

tokenizer.pad_token


# In[7]:


# tokenizer.pad_token_id = tokenizer.eos_token_id
# sft_model.lm_head

# yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
# no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
# # keep only the yes and no tokens from lm_head
# par = torch.nn.Parameter(torch.vstack([model.lm_head.weight[yes_token_id, :], model.lm_head.weight[no_token_id, :]]))
# print(par.shape)
# print(model.lm_head.weight.shape)
# model.lm_head.weight = par


# In[8]:


EOS_TOKEN = tokenizer.eos_token  # QWEN_EOS
BOS_USER_TOKEN = QWEN_BOS_USER
BOS_ASSISTANT_TOKEN = QWEN_BOS_ASSISTANT


def create_summary_cols(example):
#     example[
#         'summary_prompts_1'
#     ] = f'''{BOS_USER_TOKEN}
# Summarize the following reddit post:
# {example["post"]}{EOS_TOKEN}
# {BOS_ASSISTANT_TOKEN}
# Summary: {example["candidate_summary_1"]}{EOS_TOKEN}'''
#     example[
#         'summary_prompts_2'
#     ] = f'''{BOS_USER_TOKEN}
# Summarize the following reddit post:
# {example["post"]}{EOS_TOKEN}
# {BOS_ASSISTANT_TOKEN}
# Summary: {example["candidate_summary_2"]}{EOS_TOKEN}'''
    example[
        'summary_prompts_1'
    ] = f'''Summarize the following reddit post:
{example["post"]}
Summary: {example["candidate_summary_1"]}{EOS_TOKEN}'''

    example[
        'summary_prompts_2'
    ] = f'''Summarize the following reddit post:
{example["post"]}
Summary: {example["candidate_summary_2"]}{EOS_TOKEN}'''
    return example


comparison_train_dataset = comparison_train_dataset.map(create_summary_cols)

# tokenized_train_dataset['train']['summary_prompts_1'][0]

HF_BASELINE_RUN = False


def prepare_for_reward_modelling(example, hf_baseline: bool = HF_BASELINE_RUN):
    choice_column = example["choice"] if hf_baseline is True else example["ai_choice"]
    # ai_choice is based on index choice 0 ==summary 1, choice 1 == summary 2
    example["accepted_summary"] = (
        example['summary_prompts_2']
        if choice_column == example["constant_col"]
        else example['summary_prompts_1']
    )
    example["rejected_summary"] = (
        example['summary_prompts_1']
        if choice_column == example["constant_col"]
        else example['summary_prompts_2']
    )
    return example


comparison_train_dataset = comparison_train_dataset.map(prepare_for_reward_modelling)


# In[9]:


comparison_train_dataset


# ### Data preparation
# Encoder-Decoder specific line 29<br>
# Sampling from a range of start and end prompts could help robustness in the RM model.

# In[10]:


def tokenize_function(example, hf_baseline: bool = HF_BASELINE_RUN):
    choice_column = example["choice"] if hf_baseline is True else example["ai_choice"]
    # start_prompt = "Summarize the following reddit post.\n\n"
    # end_prompt = "\n\nSummary: "
    # prompt = [start_prompt + dialogue + end_prompt for dialogue in example["post"]]
    accepted = tokenizer(
        example["accepted_summary"],
        padding=True,
        # padding='max_length',
        truncation=True,
        return_tensors="pt",
    ).to(torch.device(DEVICE))
    example["input_ids_chosen"] = accepted.input_ids
    example["attention_mask_chosen"] = accepted.attention_mask
    rejected = tokenizer(
        example["rejected_summary"],
        padding=True,
        # padding='max_length',
        truncation=True,
        return_tensors="pt",
    ).to(torch.device(DEVICE))
    example["input_ids_rejected"] = rejected.input_ids
    example["attention_mask_rejected"] = rejected.attention_mask

    example["labels"] = tokenizer(
        [str(choice) for choice in choice_column],
        padding=True,
        # padding='max_length',
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(torch.device(DEVICE))

    # if (
    #     'causal' in sft_model.config.architectures[0].lower()
    #     and sft_model.config.architectures[0] != 'PhiForCausalLM'
    # ):
    #     example["decoder_input_ids"] = sft_model._shift_right(example["labels"])
    return example


reward_modelling_train_dataset = comparison_train_dataset.map(
    tokenize_function, batched=True
)
reward_modelling_train_dataset = reward_modelling_train_dataset.remove_columns(
    [
        'subreddit',
        'post',
        'choice',
        'candidate_summary_1',
        'candidate_summary_2',
        # 'prompts',
        'ai_choice',
        'constant_col',
        'accepted_summary',
        'rejected_summary',
        'summary_prompts_2',
        'summary_prompts_1',
    ]  # 'is_match',
)
reward_modelling_train_dataset


# The below model is the pretrained SFT model with an additional head for classification.

# In[11]:


rm_peft_model = AutoModelForSequenceClassification.from_pretrained(
    SFT_PEFT_ADAPTER_PATH, load_in_4bit=True  # torch_dtype=PRECISION
)

lora_config = LoraConfig(
    # Determines the size of LoRA matrices. x*r * r*y = x*y
    r=RM_LORA_PARAM_R,
    # scaling coefficient. Paper mentions it is important because the adjustments are small compared
    # to the rest of the model.
    lora_alpha=RM_LORA_PARAM_ALPHA,
    # Variable target_modules determines what layers are fine-tuned, see architecture above.
    # Simplest case scenario based on the original paper.
    # The parameters / layers of the new head need to be enabled for training.
    target_modules=RM_LORA_PARAM_TARGET_MODULES,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# rm_peft_model = FastLanguageModel.get_peft_model(
#     base_reward_model,
#     r=RM_LORA_PARAM_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules=RM_LORA_PARAM_TARGET_MODULES,
#     lora_alpha=RM_LORA_PARAM_ALPHA,
#     lora_dropout=0,  # Supports any, but = 0 is optimized
#     bias="none",  # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
#     random_state=42,
#     use_rslora=False,  # We support rank stabilized LoRA
#     loftq_config=None,  # And LoftQ
# )

rm_peft_model = get_peft_model(rm_peft_model, lora_config)
print(print_number_of_trainable_model_parameters(rm_peft_model))
# base_reward_model


# In[12]:


rm_peft_model


# In[13]:


reward_modelling_train_dataset['train']
# print(tokenizer.eos_token_id)
rm_peft_model.config.pad_token_id = tokenizer.eos_token_id
rm_peft_model.config.pad_token = tokenizer.pad_token
# base_reward_model.config


# In[14]:


training_args = RewardConfig(
    output_dir=RM_OUTPUT_DIR,
    # auto_find_batch_size=True,
    per_device_train_batch_size=RM_TRAIN_BATCH_SIZE,
    save_steps=500,
    learning_rate=RM_LEARNING_RATE,  # Higher learning rate than full fine-tuning.
    logging_steps=1,
    max_steps=len(reward_modelling_train_dataset['train']) // RM_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=4,
    # Use num_train_epochs = 1, warmup_ratio for full training runs!
    warmup_steps=20,
    # fp16= not is_bfloat16_supported(),
    # bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
)

reward_trainer = RewardTrainer(  # trainer class child
    model=rm_peft_model,
    args=training_args,  # trainerarguments child
    tokenizer=tokenizer,
    train_dataset=reward_modelling_train_dataset['train'],
    # num_labels=1, # Regression
)

# for _, param in peft_model.named_parameters():
#         # all_model_params += param.numel()
#         param.requires_grad = True
# print(print_number_of_trainable_model_parameters(peft_model))


# In[15]:


# rm_peft_model._has_no_labels = None


# In[16]:


start = time.time()

reward_trainer.train()
end = time.time()

duration = end - start
print(end)
print(f"Training for 1 epoch took {round(duration, 2)} seconds to execute.")

reward_trainer.model.save_pretrained(RM_PEFT_ADAPTER_PATH)
tokenizer.save_pretrained(RM_PEFT_ADAPTER_PATH)


# ### Reload and evaluate RM model

# ### Merge and save RM model (with base model)
# So that, the sentiment pipe warning is eliminated.

# In[ ]:


# base_reward_model.active_adapter


# In[ ]:


# Merging and saving the model which is trained all the way (i.e., utilising all of the data).
from peft import AutoPeftModelForSequenceClassification

merged_rm_model = AutoPeftModelForSequenceClassification.from_pretrained(
    RM_PEFT_ADAPTER_PATH, device_map='auto', torch_dtype=PRECISION
)

merged_rm_model = merged_rm_model.merge_and_unload()
merged_rm_model.save_pretrained(RM_PEFT_MERGED_MODEL_PATH)

merged_rm_model  # .to(torch.device(DEVICE))
RM_PEFT_MERGED_MODEL_PATH


# In[ ]:


### Loop this over [base, 500, 1000, 5000, 10_000, 50_000, 100_000] to calculate the pairwise accuracy


# peft_rm_model = PeftModel.from_pretrained(base_reward_model,
#                                        f'{rm_peft_model_path}',
#                                        torch_dtype=PRECISION,
#                                        is_trainable=False)

# peft_rm_model.to(torch.device('mps'))
# print('PEFT trained RM is loaded.')


# ### _Comparing HF and AIF data (RM model generated)_
# Utilising the train test split function to select a random 15% of the validation set to compare HF and AIF labels.

# In[ ]:


rm_eval_data = comparison_dataset['validation'].train_test_split(
    test_size=0.15, seed=RANDOM_SEED  # roughly 1'200 examples with 2%
)
rm_eval_data['test']
print(len(rm_eval_data['test']))

merged_rm_model.config.pad_token_id = tokenizer.eos_token_id
merged_rm_model.config.pad_token = tokenizer.pad_token


def get_rm_probabilities(
    col: str, rm_model_to_evaluate=merged_rm_model, batch_size: int = 10
) -> List[List[str]]:
    candidate_probabilities = []
    for i in range(0, len(rm_eval_data['test']), batch_size):
        candidate = rm_model_to_evaluate(
            tokenizer(
                rm_eval_data['test'][col][i : i + batch_size],
                # padding="max_length",
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(DEVICE)
        )

        candidate_probabilities.extend(candidate.logits.softmax(dim=-1).tolist())
    return candidate_probabilities


candidate_1_probabilities = get_rm_probabilities('candidate_summary_1')
candidate_2_probabilities = get_rm_probabilities('candidate_summary_2')
print(len(candidate_1_probabilities), len(candidate_2_probabilities))


# In[ ]:


# Taking the first item in probabilities will yield the probability of being a chosen summary.
rm_eval_data['test'] = rm_eval_data['test'].add_column(
    name="candidate_1_preference_probability",
    column=[item[0] for item in candidate_1_probabilities],
)
rm_eval_data['test'] = rm_eval_data['test'].add_column(
    name="candidate_2_preference_probability",
    column=[item[0] for item in candidate_2_probabilities],
)


def get_rm_labels(example):
    example["rm_choice"] = (
        0
        if example["candidate_1_preference_probability"]
        >= example["candidate_2_preference_probability"]
        else 1
    )
    example["is_match"] = 1 if example["choice"] == example["rm_choice"] else 0
    return example


# Apply the function to each example in the dataset
rm_eval_data['test'] = rm_eval_data['test'].map(get_rm_labels)

# Calculate the mean value of the 'is_match' feature
rm_aggreement_mean_value = np.round(np.mean(rm_eval_data['test']["is_match"]) * 100, 2)
print(
    f"The Reward Model (RM) is in agreement with the annotator provided labels: {rm_aggreement_mean_value}% of the times."
)
rm_eval_data


# In[ ]:


evaluator = AILabelEvaluator(
    data_to_evaluate=rm_eval_data,
    run_id=OUTPUT_RUN_ID,  # RM_TRAIN_DATA_RUN_ID if RM_TRAIN_DATA_RUN_ID is not None else RUN_ID,
)

evaluator.compute_metrics(data_split='test', predicted_col='rm_choice')


# ## Reinforcement Learning


# In[ ]:
#
# dataset = load_from_disk(SFT_DATA_OUTPUT_PATH)
# dataset
#
#
# # In[ ]:
#
#
# if TESTING is True:
#     dataset = dataset.filter(
#         lambda example, index: index % 4680 == 0, with_indices=True
#     )
# dataset
#
#
# # In[ ]:
#
#
# rl_lora_config = LoraConfig(
#     r=RL_LORA_PARAM_R,  # Rank
#     lora_alpha=RL_LORA_PARAM_ALPHA,
#     target_modules=RL_LORA_PARAM_TARGET_MODULES,
#     lora_dropout=0.05,
#     bias="none",
#     task_type=get_task_type(model=sft_model),
# )
#
#
# rl_peft_model = get_peft_model(sft_model, rl_lora_config)
# print(print_number_of_trainable_model_parameters(rl_peft_model))
#
#
# # In[ ]:
#
#
# # A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning
# loading_class = (
#     AutoModelForCausalLMWithValueHead
#     if 'causal' in sft_model.config.architectures[0].lower()
#     else AutoModelForSeq2SeqLMWithValueHead
# )
# ppo_model = loading_class.from_pretrained(
#     rl_peft_model, torch_dtype=PRECISION, is_trainable=True, device_map='auto'
# )
#
# print(f'PPO model has {print_number_of_trainable_model_parameters(ppo_model)}\n')
# print(ppo_model)
# ppo_model  # .to(torch.device(DEVICE))
#
#
# # The below function could also be adapted to sample from a variety of prompts to improve exploration and improve
# # robustness.
#
# # In[ ]:
#
#
# def tokenize_for_rl(sample):
#     # Wrap each dialogue with the instruction.
#     prompt = f"""
# Summarize the following reddit post.
#
# {sample["post"]}
#
# Summary:
# """
#     sample["input_ids"] = tokenizer.encode(prompt)
#
#     # Requirement for PPO library.
#     sample["query"] = tokenizer.decode(sample["input_ids"])
#     return sample
#
#
# # Tokenize each dialogue.
# dataset = dataset.map(tokenize_for_rl, batched=False)
# dataset.set_format(type="torch")
# dataset
#
#
# # In[ ]:
#
#
# # dataset['train']['query'][:5]
#
#
# # In[ ]:
#
#
# ref_model = create_reference_model(ppo_model)
# ref_model  # .to(torch.device(DEVICE))
# print(
#     f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n'
# )
#
#
# # In[ ]:
#
#
# def collator(data):
#     return dict((key, [d[key] for d in data]) for key in data[0])
#
#
# test_data = [
#     {"key1": "value1", "key2": "value2", "key3": "value3"},
#     {"key1": "value2", "key2": "value3", "key3": "value4"},
# ]
# print(f'Collator input: {test_data}')
# print(f'Collator output: {collator(test_data)}')
#
#
# # In[ ]:
#
#
# config = PPOConfig(
#     # Name of model to use - used only for tracking purposes
#     model_name=CHOSEN_MODEL,
#     learning_rate=RL_LEARNING_RATE,
#     ppo_epochs=RL_N_EPOCHS,
#     mini_batch_size=RL_TRAIN_MINI_BATCH_SIZE,
#     batch_size=RL_TRAIN_BATCH_SIZE,
# )
#
# ppo_trainer = PPOTrainer(
#     config=config,
#     model=ppo_model,
#     ref_model=ref_model,
#     tokenizer=tokenizer,
#     dataset=dataset["train"],
#     data_collator=collator,
# )
#
#
# # In[ ]:
#
#
# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=merged_rm_model,
#     tokenizer=tokenizer,  # device=DEVICE
# )
#
#
# # In[ ]:
#
#
# merged_rm_model.config
#
#
# # In[ ]:
#
#
# ppo_model.config.pad_token_id = tokenizer.pad_token_id
# ppo_model.config.pad_token = tokenizer.pad_token
# merged_rm_model.config.pad_token_id = tokenizer.pad_token_id
# merged_rm_model.config.pad_token = tokenizer.pad_token
#
#
# # In[ ]:
#
#
# output_min_length = 100
# output_max_length = 600
# output_length_sampler = LengthSampler(output_min_length, output_max_length)
#
# preferred_summary_index = 0
#
# generation_kwargs = {
#     "min_length": 5,
#     "temperature": 0.6,
#     "do_sample": False,
# }  # "top_k": 0.0, "top_p": 1.0 # "do_sample": True
#
# reward_kwargs = {
#     "top_k": None,  # Return all scores.
#     "function_to_apply": "none",  # Raw logits without softmax.
#     "batch_size": RL_TRAIN_BATCH_SIZE,
# }
#
# max_ppo_steps = 5
#
#
# for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     # print(step, batch)
#     # Break when you reach max_steps.
#     if step >= max_ppo_steps:
#         break
#
#     prompt_tensors = batch["input_ids"]
#
#     # Get response from FLAN-T5/PEFT LLM.
#     summary_tensors = []
#
#     for prompt_tensor in prompt_tensors:
#         max_new_tokens = output_length_sampler()
#
#         generation_kwargs["max_new_tokens"] = max_new_tokens
#         summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
#
#         summary_tensors.append(summary.squeeze()[-max_new_tokens:])
#
#     # This needs to be called "response".
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]
#
#     # Compute reward outputs.
#     query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
#     rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)
#
#     # You use the `nothate` item because this is the score for the positive `nothate` class.
#     reward_tensors = [
#         torch.tensor(reward[preferred_summary_index]["score"]) for reward in rewards
#     ]
#
#     # Run PPO step.
#     stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
#     ppo_trainer.log_stats(stats, batch, reward_tensors)
#
#     print(f'objective/kl: {stats["objective/kl"]}')
#     print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
#     print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
#     print('-'.join('' for x in range(100)))
#
#
# # ### _Evaluate RL model_
#
# # In[ ]:
#
#
# # Merging and saving the model which is trained all the way (i.e., utilising all of the data).
#
# ppo_trainer.model.save_pretrained(RM_PEFT_ADAPTER_PATH)
# tokenizer.save_pretrained(RM_PEFT_ADAPTER_PATH)
#
# merged_rm_model = AutoPeftModelForSequenceClassification.from_pretrained(
#     RM_PEFT_ADAPTER_PATH, device_map='auto', torch_dtype=PRECISION
# )
#
#
# merged_rm_model = merged_rm_model.merge_and_unload()
# merged_rm_model.save_pretrained(RM_PEFT_MERGED_MODEL_PATH)
#
# merged_rm_model  # .to(torch.device(DEVICE))
# RM_PEFT_MERGED_MODEL_PATH
#
#
# # In[ ]:
#
#
# loading_class = (
#     AutoModelForCausalLMWithValueHead
#     if 'causal' in sft_model.config.architectures[0].lower()
#     else AutoModelForSeq2SeqLMWithValueHead
# )
# ppo_model = loading_class.from_pretrained(
#     rl_peft_model, torch_dtype=PRECISION, is_trainable=True, device_map='auto'
# )
#
#
# # In[ ]:
#
#
# N_EVAL_SAMPLES = int(len(dataset['test']) * 0.2)
# print(N_EVAL_SAMPLES)
# start = time.time()
# peft_checkpoint_generation = quantitative_comparison(
#     ppo_model,
#     dataset,
#     tokenizer,
#     n_samples_to_evaluate=N_EVAL_SAMPLES,
#     batch_size=20,
#     device=DEVICE,
# )
# baseline_model_generation = quantitative_comparison(
#     sft_model,
#     dataset,
#     tokenizer,
#     n_samples_to_evaluate=N_EVAL_SAMPLES,
#     batch_size=20,
#     device=DEVICE,
# )
#
# end = time.time()
#
# duration = end - start
# print(
#     f"Evaluating N={N_EVAL_SAMPLES} samples took {round(duration, 2)} seconds to execute."
# )
#
# human_baseline_answer = dataset["test"][0:N_EVAL_SAMPLES]["summary"]
#
# zipped_summaries = list(
#     zip(human_baseline_answer, peft_checkpoint_generation, baseline_model_generation)
# )
#
#
# # In[ ]:
#
#
# # peft_checkpoint_generation
#
#
# # In[ ]:
#
#
# df = pd.DataFrame(
#     zipped_summaries,
#     columns=[
#         "human_baseline_answer",
#         "peft_checkpoint_generation",
#         "baseline_model_generation",
#     ],
# )
# df.head()
# print(df.shape)
#
#
# # In[ ]:
#
#
# rouge = evaluate.load("rouge")
#
# original_model_results = rouge.compute(
#     predictions=baseline_model_generation,
#     references=human_baseline_answer[0 : len(baseline_model_generation)],
#     use_aggregator=True,
#     use_stemmer=True,
# )
#
# peft_model_results = rouge.compute(
#     predictions=peft_checkpoint_generation,
#     references=human_baseline_answer[0 : len(peft_checkpoint_generation)],
#     use_aggregator=True,
#     use_stemmer=True,
# )
#
# original_model_results = round_dictionary_values(original_model_results)
# # instruct_model_results = round_dictionary_values(instruct_model_results)
# peft_model_results = round_dictionary_values(peft_model_results)
# print("SFT MODEL:")
# print(original_model_results)
# # print('INSTRUCT MODEL:')
# # print(instruct_model_results)
# print("PEFT MODEL:")
# print(peft_model_results)
#
#
# # In[ ]:
#
#
# COMMON_OUTPUT_PATHS = COMMON_OUTPUT_PATHS.format(
#     RUN_ID=RM_TRAIN_DATA_RUN_ID if RM_TRAIN_DATA_RUN_ID is not None else RUN_ID
# )
# METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS)
#
if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)

data_path = f'{METRICS_PATH}/rm-config.json'

# results_dict = {'sft-model': original_model_results, 'rl-model': peft_model_results}

results_dict = {
    'PRECISION_NAME': PRECISION_NAME,
    'DEVICE': DEVICE,
    'CHOSEN_MODEL': CHOSEN_MODEL,
    'RANDOM_SEED': RANDOM_SEED,
    'SFT_RUN_ID': RUN_ID,
    'RM_TRAIN_DATA_RUN_ID': RM_TRAIN_DATA_RUN_ID,
    'RM_LORA_PARAM_R': RM_LORA_PARAM_R,
    'RM_LORA_PARAM_ALPHA': RM_LORA_PARAM_ALPHA,
    'RM_LORA_PARAM_TARGET_MODULES': RM_LORA_PARAM_TARGET_MODULES,
    'RM_TRAIN_BATCH_SIZE': RM_TRAIN_BATCH_SIZE,
    'RM_LEARNING_RATE': RM_LEARNING_RATE,
    # 'RL_LORA_PARAM_R': RL_LORA_PARAM_R,
    # 'RL_LORA_PARAM_ALPHA': RL_LORA_PARAM_ALPHA,
    # 'RL_LORA_PARAM_TARGET_MODULES': RL_LORA_PARAM_TARGET_MODULES,
    # 'RL_TRAIN_BATCH_SIZE': RL_TRAIN_BATCH_SIZE,
    # 'RL_TRAIN_MINI_BATCH_SIZE': RL_TRAIN_MINI_BATCH_SIZE,
    # 'RL_LEARNING_RATE': RL_LEARNING_RATE,
    # 'RL_N_EPOCHS': RL_N_EPOCHS,
}

with open(data_path, 'w') as file:
    json.dump(results_dict, file)

print('\n\n')
print(OUTPUT_RUN_ID)
print(results_dict)
# print("Absolute percentage improvement of PPO MODEL over SFT MODEL.")
#
# improvement = np.array(list(peft_model_results.values())) - np.array(
#     list(original_model_results.values())
# )
# for key, value in zip(peft_model_results.keys(), improvement):
#     print(f'{key}: {value*100:.2f}%')


# In[ ]:


# add one more eval, for higher rated summaries.


# ## END
