"""
Simple Entry Point.

Provides a simplified interface for setup.
For complete training/eval, use the full modules.

Usage:
!git clone ...
%cd h2c-bridge
!pip install -e .
"""

# Standard imports
import torch
from h2c_bridge.config import get_default_config
from h2c_bridge.factory import H2CModelFactory  
from h2c_bridge.data.datamodule import H2CDataModule
from h2c_bridge.utils import set_seed, clear_gpu

def setup_h2c_bridge(custom_config=None):
    """Setup H2C Bridge with sensible defaults.
    
    Args:
        custom_config: Optional dictionary to override default config
        
    Returns:
        Tuple of (factory, data_module, config)
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Get config
    config = get_default_config()
    if custom_config:
        config.update(custom_config)
    
    print(">>> 1. Initializing Factory...")
    factory = H2CModelFactory(config["SHARER_ID"], config["RECEIVER_ID"])
    tok_sharer, tok_receiver = factory.load_tokenizers()
    
    print(">>> 2. Initializing Data...")
    dm = H2CDataModule(tok_sharer, tok_receiver, config)
    
    return factory, dm, config


def main():
    """Example usage."""
    # Setup
    factory, dm, config = setup_h2c_bridge()
    
    # Load models
    print(">>> 3. Loading Models...")
    sharer, receiver = factory.load_llms()
    bridge = factory.create_bridge()
    
    print("\n✓ Setup complete!")
    print(f"Sharer: {config['SHARER_ID']}")
    print(f"Receiver: {config['RECEIVER_ID']}")
    print(f"Bridge parameters: {sum(p.numel() for p in bridge.parameters()):,}")
    
    # Note: For training, you need to extract H2CEngine from the original file
    # See PACKAGE_GUIDE.md for details
    
    return factory, dm, config, sharer, receiver, bridge


if __name__ == "__main__":
    factory, dm, config, sharer, receiver, bridge = main()
