"""SpikeLink v2 — Wave-enhanced neuromorphic transport."""
from .types import V2SpikeTrain, SpikeTrainV2
from .packet import SpikelinkPacketV2, PACKET_SIZE_V2
from .codec import SpikelinkCodecV2, PrecisionAllocator
from .metrics import TransportMetrics, compute_metrics

__all__ = [
    "V2SpikeTrain",
    "SpikeTrainV2",
    "SpikelinkPacketV2",
    "PACKET_SIZE_V2",
    "SpikelinkCodecV2",
    "PrecisionAllocator",
    "TransportMetrics",
    "compute_metrics",
]
