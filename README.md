# H2C Bridge

**Hidden-to-Cache (H2C)** enables cross-architecture latent communication between LLMs by translating a large model's hidden states into KV-cache modifications for a smaller model.


## Overview

Multi-agent LLM systems typically communicate via natural language, which is both information-limited (lossy compression) and slow (autoregressive decoding). H2C bypasses this bottleneck by:

1. Extracting layer-wise hidden states from a **Sharer** model (Llama-3.1-8B)
2. Translating them via a trainable **Bridge** (dual cross-attention + FFN)
3. Injecting the result into the **Receiver**'s KV-cache (Qwen2.5-0.5B)

The Receiver then generates responses informed by the Sharer's representations—without the Sharer ever producing tokens.

## Key Results

| Method | MMLU Accuracy | Latency |
|--------|---------------|---------|
| H2C Bridge (Ours) | **45.2%** | ~0.09s |
| Receiver Only (0.5B) | 34.8% | ~0.03s |
| Text-to-Text | 33.8% | ~0.86s |
| Sharer Only (8B) | 60.2% | ~0.09s |

- **+10.4 pp** improvement over Receiver-only
- **10x faster** than Text-to-Text collaboration
- Generalizes across all 57 MMLU subject categories

## Installation

```bash
git clone https://github.com/parkerpettit/h2c-bridge.git
cd h2c-bridge
pip install -e .
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (A100 recommended)
- ~24GB VRAM (with 4-bit quantization)

## Quick Start

```python
from h2c_bridge import H2CModelFactory
from h2c_bridge.config import get_default_config
from h2c_bridge.data import H2CDataModule
from h2c_bridge.training import H2CEngine

# Setup
config = get_default_config()
factory = H2CModelFactory(config["SHARER_ID"], config["RECEIVER_ID"])
tok_s, tok_r = factory.load_tokenizers()

# Train
dm = H2CDataModule(tok_s, tok_r, config)
engine = H2CEngine(factory, dm, config)
engine.run(epochs=1)
```

Checkpoints automatically upload to WandB as versioned artifacts.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Llama-3.1-8B   │     │  Qwen2.5-0.5B   │
│    (Sharer)     │     │   (Receiver)    │
│                 │     │                 │
│  Hidden States  │     │    KV-Cache     │
│   H_l^Sharer    │     │   (K_l, V_l)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │    ┌─────────────┐    │
         └───►│  H2C Bridge │◄───┘
              │   (~59M)    │
              │             │
              │ CrossAttn   │
              │ + FFN       │
              │ + Gates     │
              └──────┬──────┘
                     │
                     ▼
              (K_l^new, V_l^new)
                     │
                     ▼
         Modified Receiver Cache
                     │
                     ▼
              Final Answer
```

## Training Data

- **80%** OpenHermes-2.5 (instruction-following)
- **20%** MMLU Auxiliary Train (multiple-choice format)
- **312,147** examples after filtering to ≤2048 tokens

## Project Structure

```
h2c_bridge/
├── models/          # Bridge architecture (projector.py, attention.py)
├── data/            # Dataset & collator
├── training/        # Trainer & engine
├── evaluation/      # MMLU evaluator, baselines
├── visualization/   # Publication-ready charts
└── config.py        # Default hyperparameters

notebooks/
├── h2c_bridge_colab.ipynb   # Training notebook used with Google Colab
```

## Evaluation

```python
from h2c_bridge.evaluation import H2CMMLUEvaluator

evaluator = H2CMMLUEvaluator(sharer, receiver, bridge, tok_r, tok_s, config)

# Run all baselines
results = evaluator.evaluate_baselines(mmlu_dataloader)

# Evaluate bridge
acc, err, latency = evaluator.evaluate_accuracy(mmlu_dataloader)
```

## Visualization

```python
from h2c_bridge.visualization import run_all_visualizations

run_all_visualizations(engine, config, themes=("dark", "light"))
```

Generates:
- Accuracy comparison charts
- Latency breakdown
- Gate distribution violin plots
- Layer-wise injection heatmaps
- Per-subject accuracy breakdown

## Citation

```bibtex
@misc{h2cbridge,
  title={Hidden-to-Cache: Latent Communication for Multi-Agent LLM Systems},
  author={Parker Pettit and Thomas Wu and Aitor Arrese-Igor and Pradesh Mainali},
  year={2025},
  note={MIT 6.4610 Final Project}
}
```

## License

MIT
