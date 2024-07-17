from typing import List

import torch
from transformers import GenerationConfig


def quantitative_comparison(
    model_to_test,
    dataset_to_test,
    tokeniser,
    device: str = None,
    x_label: str = "post",
    y_label: str = "summary",
    n_samples_to_evaluate: int = 10,
    prompt: str = None,
    max_tokens: int = 200,
    batch_size: int = 5,
) -> List[str]:
    prompt = (
        prompt
        if prompt is not None
        else """
    Summarize the following reddit post.

    {x_sample}

    Summary: """
    )
    predictions = []
    for i in range(0, n_samples_to_evaluate, batch_size):
        # Data to predict over.
        upper_limit = (
            i + batch_size
            if i + batch_size <= n_samples_to_evaluate
            else n_samples_to_evaluate
        )
        x_data = dataset_to_test["test"][i:upper_limit][x_label]
        # y_data = dataset_to_test["test"][i: upper_limit][y_label]
        # print(x_data)

        model_generations = []
        prompts = [prompt.format(x_sample=x_sample) for x_sample in x_data]
        # print(prompts)
        # Tokenize prompts in a parallel fashion
        input_ids = tokeniser(
            prompts,
            return_tensors="pt",
            padding=True,
            # truncation=True,
            # max_length=max_tokens,
        )["input_ids"]

        # Move input_ids to the appropriate device (GPU if available)
        input_ids = input_ids.to(torch.device(device))

        # Parallelised model generation
        with torch.no_grad():
            peft_model_outputs = model_to_test.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(max_new_tokens=max_tokens),
            )

        predictions.append(
            tokeniser.batch_decode(peft_model_outputs, skip_special_tokens=True)
        )

    model_generations = [
        pred for list_of_preds in predictions for pred in list_of_preds
    ]
    return model_generations
