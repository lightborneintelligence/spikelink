"""
SpikeLink adapters for neuromorphic platforms.

Adapters are lazy-loaded to avoid requiring optional dependencies
at import time. Install specific adapters with:

    pip install spikelink[neo]      # Neo / EBRAINS
    pip install spikelink[brian2]   # Brian2 simulator
    pip install spikelink[tonic]    # Event cameras
    pip install spikelink[adapters] # All adapters
"""


def __getattr__(name):
    """Lazy load adapters that require optional dependencies."""
    if name == "NeoAdapter":
        from .neo import NeoAdapter
        return NeoAdapter
    if name == "NeoAdapterV2":
        from .neo import NeoAdapterV2
        return NeoAdapterV2
    if name == "Brian2Adapter":
        from .brian2 import Brian2Adapter
        return Brian2Adapter
    if name == "Brian2AdapterV2":
        from .brian2 import Brian2AdapterV2
        return Brian2AdapterV2
    if name == "TonicAdapter":
        from .tonic import TonicAdapter
        return TonicAdapter
    if name == "TonicAdapterV2":
        from .tonic import TonicAdapterV2
        return TonicAdapterV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NeoAdapter",
    "NeoAdapterV2", 
    "Brian2Adapter",
    "Brian2AdapterV2",
    "TonicAdapter",
    "TonicAdapterV2",
]
