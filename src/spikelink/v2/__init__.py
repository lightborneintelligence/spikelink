"""
WaveML — Wave-native math primitives for SpikeLink v2.
Lightborne Intelligence

Core components:
  - WaveState: 7-shell amplitude/phase representation
  - ShellMap: Tier assignments (IDENTITY, STRUCTURE, DYNAMICS, NOISE)
  - HarmonicTransform: Signal ↔ wave domain conversion
  - ERA: Envelope-Regulated Adaptation for identity protection

These primitives enable wave-aware transport where the protocol
understands what matters (identity) vs. what can degrade (noise).
"""

from spikelink.waveml.core import (
    # Data structures
    WaveState,
    ShellMap,
    ShellTier,
    ERABounds,
    # Transforms
    HarmonicTransform,
    ERA,
    # Constants
    PHI,
    PI,
    DEFAULT_SHELL_MAP,
)

__all__ = [
    "WaveState",
    "ShellMap", 
    "ShellTier",
    "ERABounds",
    "HarmonicTransform",
    "ERA",
    "PHI",
    "PI",
    "DEFAULT_SHELL_MAP",
]
