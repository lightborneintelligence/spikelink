"""
SpikeLink ecosystem adapters.

Bridges to major neuronal simulation and analysis platforms:

    NeoAdapter            — Neo / EBRAINS electrophysiology objects
    PyNNAdapter           — PyNN simulator-independent populations
    NestAdapter           — NEST simulator spike_generator / spike_recorder
    Brian2Adapter         — Brian2 spiking neural network simulator
    TonicAdapter          — Tonic event-camera / neuromorphic datasets
    NengoAdapter          — Nengo neuromorphic simulator
    LavaAdapter           — Intel Lava / Loihi neuromorphic hardware
    SpikeInterfaceAdapter — SpikeInterface electrophysiology pipelines

All adapters use lazy imports to avoid requiring optional dependencies.
Install the extras you need:

    pip install spikelink[neo]              # Neo + quantities
    pip install spikelink[pynn]             # PyNN
    pip install spikelink[nest]             # NEST (install separately)
    pip install spikelink[brian2]           # Brian2
    pip install spikelink[tonic]            # Tonic
    pip install spikelink[nengo]            # Nengo
    pip install spikelink[lava]             # Intel Lava
    pip install spikelink[spikeinterface]   # SpikeInterface
    pip install spikelink[full]             # Everything

See docs/adapter_contract.md for the adapter specification.
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
    # Neo / EBRAINS
    "NeoAdapter",
    "NeoAdapterV2",
    # PyNN (all backends)
    "PyNNAdapter",
    "PyNNAdapterV2",
    # NEST simulator
    "NestAdapter",
    "NestAdapterV2",
    # Brian2 simulator
    "Brian2Adapter",
    "Brian2AdapterV2",
    # Tonic / event cameras
    "TonicAdapter",
    "TonicAdapterV2",
    # Nengo neuromorphic
    "NengoAdapter",
    "NengoAdapterV2",
    # Intel Lava / Loihi
    "LavaAdapter",
    "LavaAdapterV2",
    # SpikeInterface electrophysiology
    "SpikeInterfaceAdapter",
    "SpikeInterfaceAdapterV2",
]
