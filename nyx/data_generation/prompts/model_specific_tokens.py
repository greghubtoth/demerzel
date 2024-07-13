QWEN_EOS = '<|im_end|>'
QWEN_BOS_USER = '<|im_start|>user'
QWEN_BOS_ASSISTANT = '<|im_start|>assistant'
QWEN_BOS_SYSTEM = '<|im_start|>system'

PHI_EOS = '<|end|>'
PHI_BOS_USER = '<|user|>'
PHI_BOS_ASSISTANT = '<|assistant|>'

GEMMA_EOS = '<end_of_turn>'
GEMMA_BOS_USER = '<bos><start_of_turn>user'
GEMMA_BOS_ASSISTANT = '<start_of_turn>model'

# The generations will use the below suggested tokens, can be set to '' to disable, e.g. for mt0 models.
EOS_TOKEN = GEMMA_EOS
BOS_USER_TOKEN = GEMMA_BOS_USER
BOS_ASSISTANT_TOKEN = GEMMA_BOS_ASSISTANT
