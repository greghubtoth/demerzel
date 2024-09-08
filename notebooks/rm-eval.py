import json
import sys
import time

import datasets
from pathlib import Path
import argparse

from peft import PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_from_disk
import pandas as pd
import numpy as np
from nyx.evaluation import quantitative_comparison
import torch

path = Path.cwd().parent.absolute()
nyx_path = f'{path}/'
print(nyx_path)
sys.path.append(nyx_path)

from nyx.constants import (COMMON_OUTPUT_PATHS, METRICS_PATH,
                           RM_PEFT_ADAPTER_PATH, RM_PEFT_MERGED_MODEL_PATH, RL_PEFT_ADAPTER_PATH, SFT_DATA_OUTPUT_PATH, SFT_PEFT_ADAPTER_PATH)
from nyx.data_generation.prompts.model_specific_tokens import (QWEN_BOS_ASSISTANT, QWEN_BOS_USER)
parser = argparse.ArgumentParser()
parser.add_argument("rm_run_id",
                    type=str,
                    help="RM runId to evaluate the RL generations with.")
parser.add_argument("--use_qwen", action="store_true",
                    help="Evaluate qwen-7b if true otherwise phi-1-5")
args = parser.parse_args()


RM_TRAIN_DATA_RUN_ID = args.rm_run_id
USE_QWEN = args.use_qwen

# RM_TRAIN_DATA_RUN_ID = "c745c16cb14147649de39c37c90db8f5"
# USE_QWEN = False


QWEN_SFT_MODEL_ID = 'ae517069e3734bb4884dfa7fed5db18f'
CHOSEN_MODEL = "unsloth/Qwen2-7B-Instruct-bnb-4bit" if USE_QWEN is True else "microsoft/phi-1_5"
DEVICE = 'cuda'


# In[25]:


rm_common_path = COMMON_OUTPUT_PATHS.format(RUN_ID=RM_TRAIN_DATA_RUN_ID)
common_output_path = COMMON_OUTPUT_PATHS.format(RUN_ID=QWEN_SFT_MODEL_ID)


SFT_PEFT_ADAPTER_PATH = SFT_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)

RM_PEFT_ADAPTER_PATH = RM_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=rm_common_path
)
RM_PEFT_MERGED_MODEL_PATH = RM_PEFT_MERGED_MODEL_PATH.format(
    COMMON_OUTPUT_PATHS=rm_common_path
)
RL_PEFT_ADAPTER_PATH = RL_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=rm_common_path
)
METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=rm_common_path)
print(METRICS_PATH)
RM_PEFT_MERGED_MODEL_PATH


# In[26]:

BOS_USER_TOKEN = QWEN_BOS_USER
BOS_ASSISTANT_TOKEN = QWEN_BOS_ASSISTANT

if USE_QWEN is True:
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
    sft_model = AutoModelForCausalLM.from_pretrained(  # FastLanguageModel, AutoModelForCausalLM
    SFT_PEFT_ADAPTER_PATH,  # SFT_PEFT_ADAPTER_PATH,  # YOUR MODEL YOU USED FOR TRAINING
    # max_seq_length=max_seq_length,
    # dtype=dtype,
    # load_in_4bit=load_in_4bit,
    quantization_config=bnb_config,
)
    tokenizer = AutoTokenizer.from_pretrained(SFT_PEFT_ADAPTER_PATH, padding_side='left')
    dataset = load_from_disk(SFT_DATA_OUTPUT_PATH)
    dataset = dataset.filter(lambda example, index: index % 10 == 0, with_indices=True)
    
    N_EVAL_SAMPLES = int(len(dataset['test']) * 1)
    print(N_EVAL_SAMPLES)
    start = time.time()
    prompt = f"""{BOS_USER_TOKEN}
Summarize the following reddit post: """ + "{x_sample}" + """
{EOS_TOKEN}
{BOS_ASSISTANT_TOKEN}
Summary: """
    baseline_model_generation = quantitative_comparison(
        sft_model,
        dataset,
        tokenizer,
        n_samples_to_evaluate=N_EVAL_SAMPLES,
        batch_size=10,
        device=DEVICE,
        prompt=prompt, 
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
        prompt=prompt
    )
    end = time.time()
    human_baseline_answer = dataset["test"][0: N_EVAL_SAMPLES]["summary"]
else:
    prompt = f"""Summarize the following reddit post: 
    """ + "{x_sample}" + """
Summary: """
    telemetry_data_path = (
    f'{METRICS_PATH}/phi-1-5-ppo-telemetry.json'
    if CHOSEN_MODEL == "microsoft/phi-1_5"
    else f'{METRICS_PATH}/qwen2-7b-ppo-telemetry.json'
)

    with open(telemetry_data_path) as f:
        telemetry_dict = json.load(f)

    human_baseline_answer = telemetry_dict['human-baseline-answers']
    peft_checkpoint_generation = telemetry_dict['ppo-model-answers']
    baseline_model_generation = telemetry_dict['sft-model-answers']


# In[45]:


human_baseline_summary_prompts = [prompt.split('Summary:')[0] + f'Summary: {summary}' for summary, prompt in zip(human_baseline_answer, peft_checkpoint_generation)]

conversion_dataset_dict = {}

pd_df = pd.DataFrame.from_dict({'human-baseline-answers': human_baseline_summary_prompts, 'ppo-model-answers': telemetry_dict['ppo-model-answers'],
                                'sft-model-answers': telemetry_dict['sft-model-answers']})
conversion_dataset_dict['rm_test'] = datasets.Dataset.from_pandas(pd_df)

dataset_dict = datasets.DatasetDict(conversion_dataset_dict)
# split_list


# In[34]:


rm_model = AutoModelForSequenceClassification.from_pretrained(RM_PEFT_MERGED_MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(CHOSEN_MODEL, device_map="auto")

EOS_TOKEN = tokenizer.eos_token  # QWEN_EOS



# In[37]:


# telemetry_dict['human-baseline-answers'] = human_baseline_answer
# telemetry_dict['ppo-model-answers'] = peft_checkpoint_generation
# telemetry_dict['sft-model-answers'] = baseline_model_generation

# telemetry_dict['ppo-model-answers']


# In[50]:


from typing import List
import torch

tokenizer.pad_token = tokenizer.eos_token
rm_model.config.pad_token_id = tokenizer.eos_token_id
rm_model.config.pad_token = tokenizer.pad_token


def get_rl_generation_rewards(
    col: str, rm_model_to_evaluate=rm_model, batch_size: int = 2
) -> List[List[str]]:
    rewards_list = []

    

    # logits = rm_model(**tokenised_prompts).logits
    # # print(f'logits [not hate, hate]: {logits.tolist()}')
    
    # # Print the probabilities for [not hate, hate]
    # # probabilities = logits.softmax(dim=-1).tolist()[0]
    # # print(f'probabilities [not hate, hate]: {probabilities}')
    
    # # get the logits for "not hate" - this is the reward!
    preferred_answer_index = 0
    # preferred_reward = (logits[:, preferred_answer_index]).tolist()
    # print(f'reward (high): {preferred_reward}')

    
    for i in range(0, len(dataset_dict['rm_test']), batch_size):
        with torch.no_grad():
            reward_logits = rm_model_to_evaluate(
                **tokenizer(dataset_dict['rm_test'][col][i : i + batch_size], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            ).logits

        rewards_list.extend((reward_logits[:, preferred_answer_index]).tolist())
    return rewards_list

ppo_rewards = get_rl_generation_rewards('ppo-model-answers')
sft_rewards = get_rl_generation_rewards('sft-model-answers')
human_baseline_rewards = get_rl_generation_rewards('human-baseline-answers')

print(len(ppo_rewards), len(sft_rewards), len(human_baseline_rewards))

stats_to_log = {'human-baseline-answers': human_baseline_summary_prompts, 'ppo-model-answers': telemetry_dict['ppo-model-answers'],
                                'sft-model-answers': telemetry_dict['sft-model-answers']}

stats_to_log['ppo_rewards'] = ppo_rewards
stats_to_log['sft_rewards'] = sft_rewards
stats_to_log['human_baseline_rewards'] = human_baseline_rewards
stats_to_log['ppo_reward_mean'] = np.round(np.mean(ppo_rewards), 2)
stats_to_log['sft_reward_mean'] = np.round(np.mean(sft_rewards), 2)
stats_to_log['human_baseline_reward_mean'] = np.round(np.mean(human_baseline_rewards), 2)
stats_to_log['ppo_reward_std'] = np.round(np.std(ppo_rewards), 2)
stats_to_log['sft_reward_std'] = np.round(np.std(sft_rewards), 2)
stats_to_log['human_baseline_reward_std'] = np.round(np.std(human_baseline_rewards), 2)

print(np.round(np.mean(ppo_rewards), 2), np.round(np.mean(sft_rewards), 2), np.round(np.mean(human_baseline_rewards), 2))

telemetry_data_path = (
    f'{METRICS_PATH}/phi-1-5-ppo-telemetry-with-rm-scores.json'
    if CHOSEN_MODEL == "microsoft/phi-1_5"
    else f'{METRICS_PATH}/qwen2-7b-ppo-telemetry-with-rm-scores.json'
)
with open(telemetry_data_path, 'w') as file:
    json.dump(stats_to_log, file)





