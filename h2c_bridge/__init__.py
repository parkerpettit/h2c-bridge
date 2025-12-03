"""
H2C Bridge - Hidden-to-Cache Knowledge Transfer

Implements the H2C Bridge architecture for transferring knowledge from a 
larger "sharer" model to a smaller "receiver" model via cache projection.
"""

__version__ = "0.1.0"

from h2c_bridge.models.attention import H2CAttentionBlock
from h2c_bridge.models.projector import H2CProjector
from h2c_bridge.factory import H2CModelFactory
from h2c_bridge import visualization

__all__ = [
    "H2CAttentionBlock",
    "H2CProjector",
    "H2CModelFactory",
    "visualization",
]
