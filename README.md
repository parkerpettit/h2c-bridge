# H2C Bridge

Transfer knowledge from a large LLM (sharer) to a small LLM (receiver) by injecting modified key-value cache representations.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- WandB account

### Installation

```bash
# Clone repository
git clone https://github.com/parkerpettit/h2c-bridge.git
cd h2c-bridge

# Install package
pip install -e .
```


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

## Development Workflow

### Local Iteration

Edit code locally, changes auto-reload:

```bash
python -m ipython
%load_ext autoreload
%autoreload 2

from h2c_bridge import ...
```

### Colab Training

1. **Connect VSCode to Colab** ([guide](https://github.com/googlecolab/colab-vscode))
2. **Open** `notebooks/h2c_bridge_colab.ipynb`
3. **Run** notebook cells

Checkpoints save to WandB (no Drive mount needed).

## Visualization

Generate publication-ready charts:

```python
from h2c_bridge.visualization import run_all_visualizations

run_all_visualizations(engine, config, themes=("dark", "light"))
```

Outputs both presentation (dark) and publication (light) versions.

## Checkpoints

### Save

Automatic during training. Three types:
- `latest`: most recent
- `best`: highest accuracy  
- `final`: end of training

### Load

```python
# From WandB artifact
engine.load_checkpoint("entity/project/model:best")

# Or local file
engine.load_checkpoint("path/to/checkpoint.pt")
```

## Project Structure

```
h2c_bridge/
├── models/          # Bridge architecture
├── data/            # Dataset handling
├── training/        # Training loop
├── evaluation/      # MMLU benchmarks
└── visualization/   # Publication charts
```

## License

MIT
