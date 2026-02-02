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
# Core types - RELATIVE IMPORTS (note the leading dot)
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
# Adapters
from .adapters.neo import NeoAdapter
from .adapters.brian2 import Brian2Adapter
from .adapters.tonic import TonicAdapter
# Verification
from .verification.suite import VerificationSuite
from .verification.degradation import DegradationProfiler
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
    # Adapters
    "NeoAdapter",
    "Brian2Adapter",
    "TonicAdapter",
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
