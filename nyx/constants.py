# Output paths
COMMON_OUTPUT_PATHS = "./experiments/{RUN_ID}"
METRICS_PATH = "{COMMON_OUTPUT_PATHS}/metrics"

# SFT dataset
# For details on the original TL;DR dataset, see Syed et al. 2018 by Syed, Shahbaz, Voelske, Michael, Potthast, Martin,
# & Stein, Benno (2018). It is licensed under CC BY 4.0.
REDDIT_DATA_URL = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/{split}.jsonl"

# Data paths shared among experiment runs.
SFT_DATA_OUTPUT_PATH = f"./data/filtered_reddit_summarisation_data"
# SFT
SFT_OUTPUT_DIR = "{COMMON_OUTPUT_PATHS}/models/supervised-fine-tuning/events"
SFT_PEFT_MERGED_MODEL_PATH = (
    "{COMMON_OUTPUT_PATHS}/models/supervised-fine-tuning/merged-with-peft-adapter"
)
SFT_PEFT_ADAPTER_PATH = (
    "{COMMON_OUTPUT_PATHS}/models/supervised-fine-tuning/peft-checkpoint-local"
)

# Data Generation constants
# Comparison data comes from HuggingFace.
COMPARISON_DATA_PATH = "./data/restructured_openai_comparison_data"
SUMMARY_COL = "summaries"
CANDIDATE_COL = "candidate_summary"
CHOICE_COL = "choice"
TEXT_COL = "text"
INFO_COL = "info"
POST_COL = "post"
SUBREDDIT_COL = "subreddit"
PROMPTS_COL = "prompts"

# Reward Modelling paths
RM_TRAIN_DATA_PATH = "{COMMON_OUTPUT_PATHS}/data/labelled-train-data"
RM_OUTPUT_DIR = "{COMMON_OUTPUT_PATHS}/models/reward-modelling/events"
RM_PEFT_ADAPTER_PATH = (
    "{COMMON_OUTPUT_PATHS}/models/reward-modelling/peft-checkpoint-local"
)
RM_PEFT_MERGED_MODEL_PATH = (
    "{COMMON_OUTPUT_PATHS}/models/reward-modelling/merged-with-peft-adapter"
)
