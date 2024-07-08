QWEN_EOS = '<|im_end|>'
QWEN_BOS_USER = '<|im_start|>user'
QWEN_BOS_ASSISTANT = '<|im_start|>assistant'
QWEN_BOS_SYSTEM = '<|im_start|>system'

PHI_EOS = '<|end|>'
PHI_BOS_USER = '<|user|>'
PHI_BOS_ASSISTANT = '<|assistant|>'

# The generations will use the below suggested tokens, can be set to '' to disable, e.g. for mt0 models.
EOS_TOKEN = QWEN_EOS
BOS_USER_TOKEN = QWEN_BOS_USER
BOS_ASSISTANT_TOKEN = QWEN_BOS_ASSISTANT
