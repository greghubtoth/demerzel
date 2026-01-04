# Nyx - RLAIF for Text Summarization

AI MSc Dissertation Project at Bath University exploring **Reinforcement Learning from AI Feedback (RLAIF)** for Reddit text summarization.

## Overview

This project investigates automated preference labeling methods to train reward models for text summarization, comparing AI-generated labels against human judgments from OpenAI's summarization dataset.

## Pipeline

### 1. **Supervised Fine-Tuning (SFT)**
- Notebook: `01-sft-modelling-reddit-summarisation.ipynb`
- Fine-tune base models (mt0, flan-t5, phi) on Reddit TL;DR summarization task
- Uses PEFT/LoRA for efficient training

### 2. **AI Label Generation** ⭐
- Notebook: `02-ai-label-data-generation.ipynb`
- Generate preference labels by comparing summary pairs using LLMs
- **Methods**:
  - **Baseline (Lee et al.)**: Direct preference prediction with order reversal
  - **ExpeL Adaptation**: Chain-of-Thought + Reflexion + Experience Memory (RAG)
- Mitigates position bias through bidirectional comparison

### 3. **Reinforcement Learning**
- Notebook: `03-reinforcement-learning-from-ai-or-human-feedback.ipynb`
- Train reward models on AI-labeled data
- Compare RLHF vs RLAIF performance

## Key Features

| Feature | Description |
|---------|-------------|
| **Data Source** | OpenAI summarize_from_feedback (92K+ comparisons) |
| **Evaluation Axes** | Coherence, Accuracy, Coverage, Overall Quality |
| **Bias Mitigation** | Order reversal + probability averaging |
| **Evaluation Metrics** | MCC, F1, Precision, Recall, Alignment % |
| **Advanced Methods** | CoT reasoning, self-reflection, RAG-based few-shot |

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run AI label generation (testing mode)
uv run jupyter notebook notebooks/02-ai-label-data-generation.ipynb
```

## Architecture

```
nyx/
├── data_generation/
│   ├── controllers.py       # Pipeline orchestration
│   ├── data_generators.py   # Labeling method implementations
│   ├── evaluators.py        # Performance evaluation
│   └── prompts/            # Prompt templates (Lee et al., ExpeL)
├── data_loaders/           # Dataset loading abstractions
└── constants.py            # Configuration & paths
```

## Evaluation

AI-generated labels are evaluated against human preferences using:
- **Labeller Alignment**: % agreement with human labels
- **Classification Metrics**: Precision, Recall, F1-score
- **Matthews Correlation Coefficient (MCC)**: Balanced accuracy measure

## Models Tested

- `bigscience/mt0-small/large/xl`
- `google/flan-t5-small/large/xl`
- `microsoft/phi-1_5`
- `stabilityai/stablelm-2-zephyr-1_6b`

## References

- Lee et al. (2023) - RLAIF baseline methodology
- Zhao et al. (ExpeL) - Experience learning framework
- OpenAI (2020) - Learning to Summarize from Human Feedback

## License

See `LICENSE.txt`
