#!/usr/bin/env python
# coding: utf-8

# # RLHF

# ## Reward Modelling

# In[1]:


import time
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardConfig, RewardTrainer
from typing import List
import numpy as np

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline, AutoModelForCausalLM,
)
from datasets import load_from_disk

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
import uuid
from tqdm import tqdm

TESTING = True

PRECISION_NAME = 'float16'
DEVICE = "cuda"
CHOSEN_MODEL = "microsoft/phi-1_5"
# "microsoft/phi-1_5" #"bigscience/mt0-small" # "google/flan-t5-large" "stabilityai/stablelm-2-zephyr-1_6b"
RANDOM_SEED = 42
RUN_ID = uuid.uuid4().hex  # '316a2787976e4e848ea635422e2ef684'

LORA_PARAM_TARGET_MODULES = {
    "bigscience/mt0-small": ["q", "v"],
    "microsoft/phi-1_5": ["q_proj", "v_proj"],
    "microsoft/Phi-3-mini-4k-instruct": ["qkv_proj"],
    "microsoft/Phi-3-medium-4k-instruct": ["qkv_proj"],
}

RM_LORA_PARAM_R = 32
RM_LORA_PARAM_ALPHA = 64
RM_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL] + ["dense", "out_proj"]
RM_TRAIN_BATCH_SIZE = 5
RM_LEARNING_RATE = 1e-5
RM_TRAIN_DATA_RUN_ID = "06d4cf76cc9c4b2aace1dd6d208bbc01"

RL_LORA_PARAM_R = 32
RL_LORA_PARAM_ALPHA = 64
RL_LORA_PARAM_TARGET_MODULES = LORA_PARAM_TARGET_MODULES[CHOSEN_MODEL]
RL_TRAIN_BATCH_SIZE = 2
RL_TRAIN_MINI_BATCH_SIZE = 1
RL_LEARNING_RATE = 1.41e-7
RL_N_EPOCHS = 1


# In[2]:


from nyx.evaluation import quantitative_comparison
from nyx.data_generation.evaluators import AILabelEvaluator

from nyx.utils import (
    precision_enumerator,
    print_number_of_trainable_model_parameters,
    get_task_type,
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
)


common_output_path = COMMON_OUTPUT_PATHS.format(RUN_ID=RUN_ID)

SFT_PEFT_ADAPTER_PATH = SFT_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)
SFT_PEFT_MERGED_MODEL_PATH = SFT_PEFT_MERGED_MODEL_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)

RM_OUTPUT_DIR = RM_OUTPUT_DIR.format(COMMON_OUTPUT_PATHS=common_output_path)
RM_PEFT_ADAPTER_PATH = RM_PEFT_ADAPTER_PATH.format(
    COMMON_OUTPUT_PATHS=common_output_path
)
RM_PEFT_MERGED_MODEL_PATH = RM_PEFT_MERGED_MODEL_PATH.format(
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

METRICS_PATH = METRICS_PATH.format(COMMON_OUTPUT_PATHS=common_output_path)

PRECISION = precision_enumerator(PRECISION_NAME)
PRECISION


# ### Load model and data

# In[3]:


# Original dataset with train, test and validation
comparison_dataset = load_from_disk(COMPARISON_DATA_PATH)

# Train dataset with generated AI labels
comparison_train_dataset = load_from_disk(RM_TRAIN_DATA_PATH)

if TESTING is True:
    comparison_dataset = comparison_dataset.filter(
        lambda example, index: index % 4680 == 0, with_indices=True
    )
print(comparison_dataset)


# In[4]:


# sft_model = AutoModelForSeq2SeqLM.from_pretrained(
#     SFT_PEFT_MERGED_MODEL_PATH, torch_dtype=PRECISION
# )
try:
    sft_model = AutoModelForSeq2SeqLM.from_pretrained(
        SFT_PEFT_MERGED_MODEL_PATH, torch_dtype=PRECISION, device_map="auto", attn_implementation="flash_attention_2",
    )
except ValueError:
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_PEFT_MERGED_MODEL_PATH, torch_dtype=PRECISION, device_map="auto", attn_implementation="flash_attention_2"
    )

tokenizer = AutoTokenizer.from_pretrained(CHOSEN_MODEL, padding_side='left')
tokenizer.pad_token = (
    tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
)
# sft_model.to(torch.device(DEVICE))
# print("ok")
# sft_model


# In[5]:


def create_summary_cols(example):
    start_prompt = "Summarize the following reddit post.\n\n"
    end_prompt = "\n\nSummary: "
    example['summary_prompts_1'] = (
        start_prompt + example["post"] + end_prompt + example["candidate_summary_1"]
    )
    example['summary_prompts_2'] = (
        start_prompt + example["post"] + end_prompt + example["candidate_summary_2"]
    )
    return example


comparison_train_dataset = comparison_train_dataset.map(create_summary_cols)

# tokenized_train_dataset['train']['summary_prompts_1'][0]


def prepare_for_reward_modelling(example, hf_baseline: bool = False):
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


# In[6]:


print(comparison_train_dataset)


# ### Data preparation
# Encoder-Decoder specific line 29<br>
# Sampling from a range of start and end prompts could help robustness in the RM model.

# In[27]:


# In[28]:


def tokenize_function(example):
    # start_prompt = "Summarize the following reddit post.\n\n"
    # end_prompt = "\n\nSummary: "
    # prompt = [start_prompt + dialogue + end_prompt for dialogue in example["post"]]
    accepted = tokenizer(
        example["accepted_summary"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )  # .to(torch.device(DEVICE))
    example["input_ids_chosen"] = accepted.input_ids
    example["attention_mask_chosen"] = accepted.attention_mask
    rejected = tokenizer(
        example["rejected_summary"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )  # .to(torch.device(DEVICE))
    example["input_ids_rejected"] = rejected.input_ids
    example["attention_mask_rejected"] = rejected.attention_mask

    example["labels"] = tokenizer(
        [str(choice) for choice in example["ai_choice"]],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )  # .input_ids.to(torch.device(DEVICE))

    if 'causal' in sft_model.config.architectures[0].lower():
        example["decoder_input_ids"] = sft_model._shift_right(example["labels"])
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
        'prompts',
        'ai_choice',
        'constant_col',
        'accepted_summary',
        'rejected_summary',
        'summary_prompts_2',
        'summary_prompts_1',
    ]  # 'is_match',
)
print(reward_modelling_train_dataset)


# The below model is the pretrained SFT model with an additional head for classification.

# In[29]:


base_reward_model = AutoModelForSequenceClassification.from_pretrained(
    SFT_PEFT_MERGED_MODEL_PATH, torch_dtype=PRECISION
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
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

rm_peft_model = get_peft_model(base_reward_model, lora_config)
print(print_number_of_trainable_model_parameters(rm_peft_model))
# base_reward_model


# In[30]:


reward_modelling_train_dataset['train']


# In[31]:


training_args = RewardConfig(
    output_dir=RM_OUTPUT_DIR,
    # auto_find_batch_size=True,
    per_device_train_batch_size=RM_TRAIN_BATCH_SIZE,
    save_steps=10_000,
    learning_rate=RM_LEARNING_RATE,  # Higher learning rate than full fine-tuning.
    logging_steps=1,
    max_steps=len(reward_modelling_train_dataset['train']) // RM_TRAIN_BATCH_SIZE,
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


# In[32]:


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

# In[24]:


# Merging and saving the model which is trained all the way (i.e., utilising all of the data).
merged_rm_model = rm_peft_model.merge_and_unload()
merged_rm_model.save_pretrained(RM_PEFT_MERGED_MODEL_PATH)

merged_rm_model.to(torch.device(DEVICE))
# DEVICE


# In[13]:


### Loop this over [base, 500, 1000, 5000, 10_000, 50_000, 100_000] to calculate the pairwise accuracy


# peft_rm_model = PeftModel.from_pretrained(base_reward_model,
#                                        f'{rm_peft_model_path}',
#                                        torch_dtype=PRECISION,
#                                        is_trainable=False)

# peft_rm_model.to(torch.device('mps'))
# print('PEFT trained RM is loaded.')


# ### _Comparing HF and AIF data (RM model generated)_
# Utilising the train test split function to select a random 15% of the validation set to compare HF and AIF labels.

# In[25]:


rm_eval_data = comparison_dataset['validation'].train_test_split(
    test_size=0.15, seed=RANDOM_SEED
)
rm_eval_data['test']
print(len(rm_eval_data['test']))


def get_rm_probabilities(
    col: str, rm_model_to_evaluate=merged_rm_model
) -> List[List[str]]:
    candidate = rm_model_to_evaluate(
        tokenizer(
            rm_eval_data['test'][col],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
    )

    candidate_probabilities = candidate.logits.softmax(dim=-1).tolist()
    return candidate_probabilities


candidate_1_probabilities = get_rm_probabilities('candidate_summary_1')
candidate_2_probabilities = get_rm_probabilities('candidate_summary_2')


# In[26]:


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


# In[27]:


evaluator = AILabelEvaluator(data_to_evaluate=rm_eval_data, run_id=RUN_ID)

evaluator.compute_metrics(data_split='test', predicted_col='rm_choice')


# ## Reinforcement Learning

# In[28]:

#
# dataset = load_from_disk(SFT_DATA_OUTPUT_PATH)
# dataset
#
#
# # In[29]:
#
#
# if TESTING is True:
#     dataset = dataset.filter(
#         lambda example, index: index % 4680 == 0, with_indices=True
#     )
# dataset
#
#
# # In[30]:
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
# # In[38]:
#
#
# # A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning
# ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
#     rl_peft_model, torch_dtype=PRECISION, is_trainable=True
# )
#
# print(f'PPO model has {print_number_of_trainable_model_parameters(ppo_model)}\n')
# print(ppo_model)
# ppo_model.to(torch.device(DEVICE))
#
#
# # The below function could also be adapted to sample from a variety of prompts to improve exploration and improve
# # robustness.
#
# # In[39]:
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
# # In[45]:
#
#
# ref_model = create_reference_model(ppo_model)
# ref_model.to(torch.device(DEVICE))
# print(
#     f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n'
# )
#
#
# # In[46]:
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
# # In[47]:
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
# # In[48]:
#
#
# sentiment_pipe = pipeline(
#     "sentiment-analysis", model=merged_rm_model, tokenizer=tokenizer, device=DEVICE
# )
#
#
# # In[50]:
#
#
# output_min_length = 100
# output_max_length = 3_500
# output_length_sampler = LengthSampler(output_min_length, output_max_length)
#
# preferred_summary_index = 0
#
# generation_kwargs = {"min_length": 5, "top_k": 0.0, "top_p": 1.0, "do_sample": True}
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
# N_EVAL_SAMPLES = int(len(tokenized_datasets['test']) * 0.15)
#
# start = time.time()
# peft_checkpoint_generation = quantitative_comparison(
#     ppo_model,
#     dataset,
#     tokenizer,
#     n_samples_to_evaluate=N_EVAL_SAMPLES,
#     batch_size=2,
#     device=DEVICE,
# )
# baseline_model_generation = quantitative_comparison(
#     sft_model,
#     dataset,
#     tokenizer,
#     n_samples_to_evaluate=N_EVAL_SAMPLES,
#     batch_size=2,
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


# In[ ]:


# if not os.path.exists(METRICS_PATH):
#     os.makedirs(METRICS_PATH)
#
# data_path = f'{METRICS_PATH}/rl-results.json'
#
# results_dict = {'sft-model': original_model_results, 'rl-model': peft_model_results}
# with open(data_path, 'w') as file:
#     json.dump(results_dict, file)
# print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")
#
# improvement = np.array(list(peft_model_results.values())) - np.array(
#     list(original_model_results.values())
# )
# for key, value in zip(peft_model_results.keys(), improvement):
#     print(f'{key}: {value*100:.2f}%')


# In[ ]:


# add one more eval, for higher rated summaries.


# ## END
