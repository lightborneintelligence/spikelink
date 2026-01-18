"""
Convenience API — Simple encode/decode/verify functions.

Provides one-liner access to common SpikeLink operations.
"""

from typing import List, Union

import numpy as np

from spikelink.core.codec import SpikelinkCodec
from spikelink.core.packet import SpikelinkPacket
from spikelink.types.spiketrain import SpikeTrain

# Default codec instance
_default_codec = SpikelinkCodec()


def encode(
    times: Union[List[float], np.ndarray, SpikeTrain],
) -> List[SpikelinkPacket]:
    """
    Encode spike times to packets.

    Args:
        times: Spike times as list, array, or SpikeTrain

    Returns:
        List of SpikelinkPackets

    Example:
        >>> packets = encode([0.1, 0.2, 0.3, 0.4, 0.5])
    """
    if isinstance(times, SpikeTrain):
        train = times
    else:
        train = SpikeTrain(times=times)

    return _default_codec.encode_train(train)


def decode(packets: List[SpikelinkPacket]) -> np.ndarray:
    """
    Decode packets to spike times.

    Args:
        packets: List of SpikelinkPackets

    Returns:
        Numpy array of spike times

    Example:
        >>> times = decode(packets)
    """
    train = _default_codec.decode_packets(packets)
    return train.times


def verify(
    original: Union[List[float], np.ndarray, SpikeTrain],
    recovered: Union[List[float], np.ndarray, SpikeTrain],
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify that recovered spike times match original.

    Args:
        original: Original spike times
        recovered: Recovered spike times after transport
        tolerance: Maximum allowed error per spike

    Returns:
        True if verification passes, False otherwise

    Example:
        >>> passed = verify(original_times, recovered_times)
    """
    # Convert to arrays
    if isinstance(original, SpikeTrain):
        orig_times = original.times
    else:
        orig_times = np.asarray(original)

    if isinstance(recovered, SpikeTrain):
        rec_times = recovered.times
    else:
        rec_times = np.asarray(recovered)

    # Check count
    if len(orig_times) != len(rec_times):
        return False

    # Check empty case
    if len(orig_times) == 0:
        return True

    # Check values
    max_error = np.max(np.abs(orig_times - rec_times))
    return max_error <= tolerance


def round_trip(
    times: Union[List[float], np.ndarray, SpikeTrain],
) -> np.ndarray:
    """
    Perform encode-decode round trip.

    Useful for testing what data looks like after transport.

    Args:
        times: Original spike times

    Returns:
        Spike times after round trip

    Example:
        >>> recovered = round_trip([0.1, 0.2, 0.3])
    """
    packets = encode(times)
    return decode(packets)
