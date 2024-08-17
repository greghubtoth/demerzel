#!/usr/bin/env python
# coding: utf-8


import time
import json
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
from trl import RewardConfig, RewardTrainer
from typing import List
import numpy as np
import pandas as pd
import evaluate

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from datasets import load_from_disk

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForSeq2SeqLMWithValueHead,
    AutoModelForCausalLMWithValueHead,
)
from trl import create_reference_model
from trl.core import LengthSampler
import uuid
from tqdm import tqdm

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

# RM_LORA_PARAM_R = 16
# RM_LORA_PARAM_ALPHA = 16
# RM_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL] + [
#     # "dense",
#     # "out_proj",
#     "score"
# ]
# RM_TRAIN_BATCH_SIZE = 5
# RM_LEARNING_RATE = 5e-5
RM_TRAIN_DATA_RUN_ID = "bbea9d66b20148a29fa41ff1cc586558"
print(f"RM_TRAIN_DATA_RUN_ID: {RM_TRAIN_DATA_RUN_ID}")

RL_LORA_PARAM_R = 16
RL_LORA_PARAM_ALPHA = 16
RL_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL]
RL_TRAIN_BATCH_SIZE = 25
RL_TRAIN_MINI_BATCH_SIZE = 5
RL_LEARNING_RATE = 1.41e-5
RL_N_EPOCHS = 1


# In[3]:


import sys
from pathlib import Path

path = Path.cwd().parent.absolute()
nyx_path = f'{path}/'
print(nyx_path)
sys.path.append(nyx_path)


# In[4]:


from nyx.evaluation import quantitative_comparison
from nyx.data_generation.evaluators import AILabelEvaluator

from nyx.utils import (
    precision_enumerator,
    print_number_of_trainable_model_parameters,
    get_task_type,
    round_dictionary_values,
)
from nyx.constants import (
    COMPARISON_DATA_PATH,
    SFT_DATA_OUTPUT_PATH,
    COMMON_OUTPUT_PATHS,
    SFT_PEFT_MERGED_MODEL_PATH,
    SFT_PEFT_ADAPTER_PATH,
    RM_TRAIN_DATA_PATH,
    RM_OUTPUT_DIR,
    RM_PEFT_ADAPTER_PATH,
    RM_PEFT_MERGED_MODEL_PATH,
    METRICS_PATH,
    RL_OUTPUT_DIR,
    RL_PEFT_ADAPTER_PATH,
    RL_PEFT_MERGED_MODEL_PATH,
)
from nyx.data_generation.prompts.model_specific_tokens import (
    QWEN_EOS,
    QWEN_BOS_USER,
    QWEN_BOS_ASSISTANT,
)

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
    RM_OUTPUT_DIR = RM_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=rm_common_path)
    RM_PEFT_ADAPTER_PATH = RM_PEFT_ADAPTER_PATH.format(
        COMMON_OUTPUT_PATHS=rm_common_path
    )
    RM_PEFT_MERGED_MODEL_PATH = RM_PEFT_MERGED_MODEL_PATH.format(
        COMMON_OUTPUT_PATHS=rm_common_path
    )
    RL_OUTPUT_DIR = RL_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=rm_common_path)
    RL_PEFT_ADAPTER_PATH = RL_PEFT_ADAPTER_PATH.format(COMMON_OUTPUT_PATHS=rm_common_path)
    RL_PEFT_MERGED_MODEL_PATH = RL_PEFT_MERGED_MODEL_PATH.format(COMMON_OUTPUT_PATHS =rm_common_path)
else:  # utilise current run_id
    RM_TRAIN_DATA_PATH = RM_TRAIN_DATA_PATH.format(
        COMMON_OUTPUT_PATHS=common_output_path
    )
    RM_OUTPUT_DIR = RM_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=common_output_path)
    RM_PEFT_ADAPTER_PATH = RM_PEFT_ADAPTER_PATH.format(
        COMMON_OUTPUT_PATHS=common_output_path
    )
    RM_PEFT_MERGED_MODEL_PATH = RM_PEFT_MERGED_MODEL_PATH.format(
        COMMON_OUTPUT_PATHS=common_output_path
    )

METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=common_output_path)

PRECISION = precision_enumerator(PRECISION_NAME)
PRECISION


# ### Load model and data

# In[5]:


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
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

sft_model = AutoModelForCausalLM.from_pretrained( # FastLanguageModel, AutoModelForCausalLM
    SFT_PEFT_ADAPTER_PATH, #SFT_PEFT_ADAPTER_PATH,  # YOUR MODEL YOU USED FOR TRAINING
    # max_seq_length=max_seq_length,
    # dtype=dtype,
    # load_in_4bit=load_in_4bit,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(SFT_PEFT_ADAPTER_PATH, padding_side='left')
tokenizer.pad_token = (
    tokenizer.pad_token
    if tokenizer.pad_token is not None
    else tokenizer.eos_token  # '<|endoftext|>'
)
tokenizer.padding_side = 'left'

tokenizer.pad_token


# In[6]:


EOS_TOKEN = QWEN_EOS
BOS_USER_TOKEN = QWEN_BOS_USER
BOS_ASSISTANT_TOKEN = QWEN_BOS_ASSISTANT


# ### Reload and evaluate RM model

# In[7]:


# base_reward_model.active_adapter
RM_PEFT_ADAPTER_PATH


# In[8]:


# Merging and saving the model which is trained all the way (i.e., utilising all of the data).
from peft import AutoPeftModelForSequenceClassification
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

merged_rm_model = AutoPeftModelForSequenceClassification.from_pretrained(
    RM_PEFT_ADAPTER_PATH, quantization_config=bnb_config, device_map=DEVICE
)
merged_rm_model = merged_rm_model.merge_and_unload()

merged_rm_model  # .to(torch.device(DEVICE))
RM_PEFT_MERGED_MODEL_PATH


# ## Reinforcement Learning

# In[9]:


dataset = load_from_disk(SFT_DATA_OUTPUT_PATH)
dataset


# In[10]:


if TESTING is True:
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(30))
    dataset["validation"] = dataset["validation"].select(
        range(50)
    )
else:
    dataset = dataset.filter(
        lambda example, index: index % 10 == 0, with_indices=True
    )
dataset


# In[11]:


rl_lora_config = LoraConfig(
    r=RL_LORA_PARAM_R,  # Rank
    lora_alpha=RL_LORA_PARAM_ALPHA,
    target_modules=RL_LORA_PARAM_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=get_task_type(model=sft_model),
)


rl_peft_model = get_peft_model(sft_model, rl_lora_config)
print(print_number_of_trainable_model_parameters(rl_peft_model))


# In[12]:


# A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning
loading_class = (
    AutoModelForCausalLMWithValueHead
    if 'causal' in sft_model.config.architectures[0].lower()
    else AutoModelForSeq2SeqLMWithValueHead
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

ppo_model = loading_class.from_pretrained(
    rl_peft_model, is_trainable=True, quantization_config=bnb_config, device_map=DEVICE
)

print(f'PPO model has {print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model)
ppo_model  # .to(torch.device(DEVICE))


# The below function could also be adapted to sample from a variety of prompts to improve exploration and improve
# robustness.

# In[24]:


def tokenize_for_rl(sample):
    # Wrap each dialogue with the instruction.
    prompt = f"""
Summarize the following reddit post.

{sample["post"]}

Summary:
"""
    sample["input_ids"] = tokenizer.encode(prompt)
    # sample['countThis'] = tokenizer.encode(sample['summary'])
    # Requirement for PPO library.
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


# Tokenize each dialogue.
dataset = dataset.map(tokenize_for_rl, batched=False)
dataset.set_format(type="torch")
dataset


# In[25]:


# import numpy as np
# y = [len(x) for x in dataset['train']['countThis']]
# np.mean(y), np.min(y), np.max(y)


# In[26]:


ref_model = create_reference_model(ppo_model)
ref_model  # .to(torch.device(DEVICE))
print(
    f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n'
)


# In[27]:


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


test_data = [
    {"key1": "value1", "key2": "value2", "key3": "value3"},
    {"key1": "value2", "key2": "value3", "key3": "value4"},
]
print(f'Collator input: {test_data}')
print(f'Collator output: {collator(test_data)}')


# In[28]:


config = PPOConfig(
    # Name of model to use - used only for tracking purposes
    model_name=CHOSEN_MODEL,
    learning_rate=RL_LEARNING_RATE,
    ppo_epochs=RL_N_EPOCHS,
    mini_batch_size=RL_TRAIN_MINI_BATCH_SIZE,
    batch_size=RL_TRAIN_BATCH_SIZE,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset["train"],
    data_collator=collator,
)


# In[29]:


sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=merged_rm_model,
    tokenizer=tokenizer,  # device=DEVICE
)


# In[30]:


merged_rm_model.config


# In[31]:


ppo_model.config.pad_token_id = tokenizer.pad_token_id
ppo_model.config.pad_token = tokenizer.pad_token
merged_rm_model.config.pad_token_id = tokenizer.pad_token_id
merged_rm_model.config.pad_token = tokenizer.pad_token


# In[32]:


output_min_length = 25
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

preferred_summary_index = 0

generation_kwargs = {
    "min_length": 5,
    "temperature": 0.6,
    "do_sample": False,
}  # "top_k": 0.0, "top_p": 1.0 # "do_sample": True

reward_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "none",  # Raw logits without softmax.
    "batch_size": RL_TRAIN_BATCH_SIZE,
}

max_ppo_steps = dataset['train'].num_rows // RL_TRAIN_BATCH_SIZE
print(f'Max steps are: {max_ppo_steps}.')

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # print(step, batch)
    print(step, max_ppo_steps)
    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    # Get response from FLAN-T5/PEFT LLM.
    summary_tensors = []

    # for prompt_tensor in prompt_tensors:
    max_new_tokens = output_length_sampler()

    generation_kwargs["max_new_tokens"] = max_new_tokens
    #     summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

    #     summary_tensors.append(summary.squeeze()[-max_new_tokens:])

    response_tensors = ppo_trainer.generate(prompt_tensors, **generation_kwargs)

    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute reward outputs.
    # query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = sentiment_pipe(batch["response"], **reward_kwargs)

    # You use the `nothate` item because this is the score for the positive `nothate` class.
    reward_tensors = [
        torch.tensor(reward[preferred_summary_index]["score"]) for reward in rewards
    ]

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-'.join('' for x in range(100)))


# In[ ]:


# batch["response"]


# ### _Evaluate RL model_

# In[ ]:


# RL_OUTPUT_DIR = "./experiments/8e9a5881d6c04fb6b20042829c9e5c3c/models/reinforcement-learning/peft-checkpoint-local"


# In[33]:


# Merging and saving the model which is trained all the way (i.e., utilising all of the data).
if not os.path.exists(RL_OUTPUT_DIR):
    os.makedirs(RL_OUTPUT_DIR)
ppo_trainer.model.save_pretrained(RL_PEFT_ADAPTER_PATH)
tokenizer.save_pretrained(RL_PEFT_ADAPTER_PATH)

# merged_rl_model = AutoPeftModelForSequenceClassification.from_pretrained(
#     RL_PEFT_ADAPTER_PATH, device_map='auto', #torch_dtype=PRECISION
# )


# ppo_model = merged_rl_model.merge_and_unload()
# ppo_model.save_pretrained(RL_PEFT_MERGED_MODEL_PATH)

ppo_model  # .to(torch.device(DEVICE))
# RL_PEFT_MERGED_MODEL_PATH


# In[34]:


RL_PEFT_ADAPTER_PATH


# In[35]:


# loading_class = (
#     AutoModelForCausalLMWithValueHead
#     if 'causal' in sft_model.config.architectures[0].lower()
#     else AutoModelForSeq2SeqLMWithValueHead
# )
# ppo_model = loading_class.from_pretrained(
#     rl_peft_model, torch_dtype=PRECISION, is_trainable=False, device_map='auto'
# )


# In[36]:


N_EVAL_SAMPLES = int(len(dataset['test']) * 1)
print(N_EVAL_SAMPLES)
start = time.time()
baseline_model_generation = quantitative_comparison(
    sft_model,
    dataset,
    tokenizer,
    n_samples_to_evaluate=N_EVAL_SAMPLES,
    batch_size=10,
    device=DEVICE,
)

peft_config = PeftConfig.from_pretrained(RL_PEFT_ADAPTER_PATH)
# to initiate with random weights
peft_config.init_lora_weights = False
sft_model.add_adapter(peft_config, adapter_name='ppo')
sft_model.enable_adapters()
sft_model

peft_checkpoint_generation = quantitative_comparison(
    sft_model,
    dataset,
    tokenizer,
    n_samples_to_evaluate=N_EVAL_SAMPLES,
    batch_size=10,
    device=DEVICE,
)
end = time.time()

duration = end - start
print(
    f"Evaluating N={N_EVAL_SAMPLES} samples took {round(duration, 2)} seconds to execute."
)

human_baseline_answer = dataset["test"][0: N_EVAL_SAMPLES]["summary"]

zipped_summaries = list(
    zip(human_baseline_answer, peft_checkpoint_generation, baseline_model_generation)
)


# In[37]:


# peft_checkpoint_generation


# In[38]:


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


# In[39]:


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
print("SFT MODEL:")
print(original_model_results)
# print('INSTRUCT MODEL:')
# print(instruct_model_results)
print("PEFT MODEL:")
print(peft_model_results)


# In[40]:


COMMON_OUTPUT_PATHS = COMMON_OUTPUT_PATHS.format(
    RUN_ID=RM_TRAIN_DATA_RUN_ID if RM_TRAIN_DATA_RUN_ID is not None else RUN_ID
)
METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=COMMON_OUTPUT_PATHS)

if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)

data_path = f'{METRICS_PATH}/rl-results.json'

results_dict = {'sft-model': original_model_results, 'rl-model': peft_model_results}

with open(data_path, 'w') as file:
    json.dump(results_dict, file)

print("Absolute percentage improvement of PPO MODEL over SFT MODEL.")

improvement = np.array(list(peft_model_results.values())) - np.array(
    list(original_model_results.values())
)
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')


# In[41]:


results_dict = {
    'PRECISION_NAME': PRECISION_NAME,
    'DEVICE': DEVICE,
    'CHOSEN_MODEL': CHOSEN_MODEL,
    'RANDOM_SEED': RANDOM_SEED,
    'SFT_RUN_ID': RUN_ID,
    'RM_TRAIN_DATA_RUN_ID': RM_TRAIN_DATA_RUN_ID,
    'RL_LORA_PARAM_R': RL_LORA_PARAM_R,
    'RL_LORA_PARAM_ALPHA': RL_LORA_PARAM_ALPHA,
    'RL_LORA_PARAM_TARGET_MODULES': RL_LORA_PARAM_TARGET_MODULES,
    'RL_TRAIN_BATCH_SIZE': RL_TRAIN_BATCH_SIZE,
    'RL_TRAIN_MINI_BATCH_SIZE': RL_TRAIN_MINI_BATCH_SIZE,
    'RL_LEARNING_RATE': RL_LEARNING_RATE,
    'RL_N_EPOCHS': RL_N_EPOCHS,
    'RL_MAX_PPO_STEPS': max_ppo_steps,
}

data_path = f'{METRICS_PATH}/rl-config.json'
with open(data_path, 'w') as file:
    json.dump(results_dict, file)


# In[ ]:


print('\n\n')
print(f'RUN_ID: {RUN_ID}\nRM_TRAIN_DATA_RUN_ID: {RM_TRAIN_DATA_RUN_ID}')
print(results_dict)


# ## END
