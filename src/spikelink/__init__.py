"""
SpikeLink — Spike-native transport for neuromorphic systems.

SpikeLink transports spike symbols without forcing early binary collapse.
It operates post-binary within digital systems, preserving symbol magnitude
continuity and enabling graceful degradation under noise.

Key Properties:
- Spike-native: No ADC/DAC conversion stages
- Graceful degradation: Precision loss under noise, not data loss
- EBRAINS compatible: Validated against Neo, Elephant, PyNN workflows
- Time-coherent: Bounded timing, predictable behavior

Example:
    >>> from spikelink import SpikeTrain, SpikelinkCodec
    >>> train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
    >>> codec = SpikelinkCodec()
    >>> packets = codec.encode_train(train)
    >>> recovered = codec.decode_packets(packets)

License: Apache-2.0
Copyright (c) 2026 Lightborne Intelligence
"""

__version__ = "0.2.0"
__author__ = "Jesus Carrasco"
__license__ = "Apache-2.0"

# Convenience API
from .api import decode, encode, verify

# Core protocol (v1)
from .core.codec import SpikelinkCodec
from .core.packet import SpikelinkPacket
from .types.spiketrain import SpikeTrain

# V2 protocol
from .v2.codec import SpikelinkCodecV2
from .v2.packet import SpikelinkPacketV2
from .v2.types import SpikeTrainV2
from .v2.metrics import compute_metrics

# Verification
from .verification.suite import VerificationSuite
from .verification.degradation import DegradationProfiler


# Adapters - LAZY LOADING (optional dependencies)
def __getattr__(name):
    """Lazy load adapters that require optional dependencies."""
    # Neo / EBRAINS
    if name == "NeoAdapter":
        from .adapters.neo import NeoAdapter
        return NeoAdapter
    elif name == "NeoAdapterV2":
        from .adapters.neo import NeoAdapterV2
        return NeoAdapterV2
    
    # PyNN (all backends)
    elif name == "PyNNAdapter":
        from .adapters.pynn import PyNNAdapter
        return PyNNAdapter
    elif name == "PyNNAdapterV2":
        from .adapters.pynn import PyNNAdapterV2
        return PyNNAdapterV2
    
    # NEST simulator
    elif name == "NestAdapter":
        from .adapters.nest import NestAdapter
        return NestAdapter
    elif name == "NestAdapterV2":
        from .adapters.nest import NestAdapterV2
        return NestAdapterV2
    
    # Brian2 simulator
    elif name == "Brian2Adapter":
        from .adapters.brian2 import Brian2Adapter
        return Brian2Adapter
    elif name == "Brian2AdapterV2":
        from .adapters.brian2 import Brian2AdapterV2
        return Brian2AdapterV2
    
    # Tonic / event cameras
    elif name == "TonicAdapter":
        from .adapters.tonic import TonicAdapter
        return TonicAdapter
    elif name == "TonicAdapterV2":
        from .adapters.tonic import TonicAdapterV2
        return TonicAdapterV2
    
    # Nengo neuromorphic
    elif name == "NengoAdapter":
        from .adapters.nengo import NengoAdapter
        return NengoAdapter
    elif name == "NengoAdapterV2":
        from .adapters.nengo import NengoAdapterV2
        return NengoAdapterV2
    
    # Intel Lava / Loihi
    elif name == "LavaAdapter":
        from .adapters.lava import LavaAdapter
        return LavaAdapter
    elif name == "LavaAdapterV2":
        from .adapters.lava import LavaAdapterV2
        return LavaAdapterV2
    
    # SpikeInterface electrophysiology
    elif name == "SpikeInterfaceAdapter":
        from .adapters.spikeinterface import SpikeInterfaceAdapter
        return SpikeInterfaceAdapter
    elif name == "SpikeInterfaceAdapterV2":
        from .adapters.spikeinterface import SpikeInterfaceAdapterV2
        return SpikeInterfaceAdapterV2
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Types (v1)
    "SpikeTrain",
    # Core (v1)
    "SpikelinkCodec",
    "SpikelinkPacket",
    # V2 Protocol
    "SpikeTrainV2",
    "SpikelinkCodecV2",
    "SpikelinkPacketV2",
    "compute_metrics",
    # Adapters (lazy-loaded)
    "NeoAdapter",
    "NeoAdapterV2",
    "PyNNAdapter",
    "PyNNAdapterV2",
    "NestAdapter",
    "NestAdapterV2",
    "Brian2Adapter",
    "Brian2AdapterV2",
    "TonicAdapter",
    "TonicAdapterV2",
    "NengoAdapter",
    "NengoAdapterV2",
    "LavaAdapter",
    "LavaAdapterV2",
    "SpikeInterfaceAdapter",
    "SpikeInterfaceAdapterV2",
    # Verification
    "VerificationSuite",
    "DegradationProfiler",
    # Convenience
    "encode",
    "decode",
    "verify",
]


def get_version() -> str:
    """Return the current SpikeLink version."""
    return __version__
