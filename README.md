# H2C Bridge

A modular implementation of the H2C Bridge architecture. It transfers knowledge from a large "sharer" model (e.g., Llama-3) to a smaller "receiver" model (e.g., Qwen-0.5B) by projecting hidden states into the receiver's KV cache.

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/h2c-bridge.git
cd h2c-bridge
pip install -e .
```

### 2. Train

```python
from h2c_bridge.simple_entry import setup_h2c_bridge
from h2c_bridge.training.engine import H2CEngine

# One-line setup with defaults
factory, dm, config = setup_h2c_bridge()

# Train
engine = H2CEngine(factory, dm, config)
engine.run(epochs=1)
```

## Development Workflow

### Local Iteration (IDE + Colab)

If you're using the Colab extension in VS Code:

1. Open `notebooks/h2c_bridge_colab.ipynb`
2. Connect to your Colab runtime
3. Set `PROJECT_PATH` in the notebook to your Google Drive folder
4. **Edit code locally** in `h2c_bridge/` -> **Run cells** to test

The notebook is configured with `%autoreload`, so your code changes apply immediately without restarting the kernel.

### Visualization

We include a full visualization suite for analyzing bridge behavior.

```python
from h2c_bridge.visualization import run_all_visualizations

# Generates publication-ready charts (heatmap, radar, probability shifts)
run_all_visualizations(engine, config)
```

## Project Structure

- **`h2c_bridge/models`**: Core architecture (`H2CAttentionBlock`, `H2CProjector`)
- **`h2c_bridge/training`**: Training loop and orchestration (`Trainer`, `Engine`)
- **`h2c_bridge/evaluation`**: MMLU benchmarking and validation
- **`h2c_bridge/visualization`**: Plotting and analysis tools
- **`h2c_bridge/data`**: Dataset wrappers and collators

## Configuration

You can customize everything via the config dict:

```python
config = get_default_config()
config.update({
    "SHARER_ID": "meta-llama/Llama-3.1-8B-Instruct",
    "RECEIVER_ID": "Qwen/Qwen2.5-0.5B-Instruct",
    "BATCH_SIZE": 8,
    "lr": 1e-4
})
```

## Checkpoints

Checkpoints are saved to `drive/MyDrive/nlp/checkpoints/` by default:
- `*_best.pt`: Best validation performance
- `*_final.pt`: End of training state

## License

MIT
