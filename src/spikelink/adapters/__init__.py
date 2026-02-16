"""
SpikeLink ecosystem adapters.

Bridges to major neuronal simulation and analysis platforms.
All adapters use lazy imports to avoid requiring optional dependencies.
"""


def __getattr__(name):
    """Lazy load adapters that require optional dependencies."""
    # Neo adapters
    if name == "NeoAdapter":
        from .neo import NeoAdapter
        return NeoAdapter
    if name == "NeoAdapterV2":
        from .neo import NeoAdapterV2
        return NeoAdapterV2
    
    # PyNN adapters
    if name == "PyNNAdapter":
        from .pynn import PyNNAdapter
        return PyNNAdapter
    if name == "PyNNAdapterV2":
        from .pynn import PyNNAdapterV2
        return PyNNAdapterV2
    
    # NEST adapters
    if name == "NestAdapter":
        from .nest import NestAdapter
        return NestAdapter
    if name == "NestAdapterV2":
        from .nest import NestAdapterV2
        return NestAdapterV2
    
    # Brian2 adapters
    if name == "Brian2Adapter":
        from .brian2 import Brian2Adapter
        return Brian2Adapter
    if name == "Brian2AdapterV2":
        from .brian2 import Brian2AdapterV2
        return Brian2AdapterV2
    
    # Tonic adapters
    if name == "TonicAdapter":
        from .tonic import TonicAdapter
        return TonicAdapter
    if name == "TonicAdapterV2":
        from .tonic import TonicAdapterV2
        return TonicAdapterV2
    
    # Nengo adapters
    if name == "NengoAdapter":
        from .nengo import NengoAdapter
        return NengoAdapter
    if name == "NengoAdapterV2":
        from .nengo import NengoAdapterV2
        return NengoAdapterV2
    
    # Lava adapters
    if name == "LavaAdapter":
        from .lava import LavaAdapter
        return LavaAdapter
    if name == "LavaAdapterV2":
        from .lava import LavaAdapterV2
        return LavaAdapterV2
    
    # SpikeInterface adapters
    if name == "SpikeInterfaceAdapter":
        from .spikeinterface import SpikeInterfaceAdapter
        return SpikeInterfaceAdapter
    if name == "SpikeInterfaceAdapterV2":
        from .spikeinterface import SpikeInterfaceAdapterV2
        return SpikeInterfaceAdapterV2
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NeoAdapter", "NeoAdapterV2",
    "PyNNAdapter", "PyNNAdapterV2",
    "NestAdapter", "NestAdapterV2",
    "Brian2Adapter", "Brian2AdapterV2",
    "TonicAdapter", "TonicAdapterV2",
    "NengoAdapter", "NengoAdapterV2",
    "LavaAdapter", "LavaAdapterV2",
    "SpikeInterfaceAdapter", "SpikeInterfaceAdapterV2",
]
