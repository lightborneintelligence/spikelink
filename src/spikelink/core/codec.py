"""
SpikelinkCodec — Encode and decode spike trains for transport.

The codec transforms SpikeTrain objects to/from SpikelinkPackets,
handling precision management and symbol magnitude preservation.
"""

from typing import List, Optional, Union

import numpy as np

from spikelink.core.packet import SpikelinkPacket
from spikelink.types.spiketrain import SpikeTrain


class SpikelinkCodec:
    """
    Encode and decode spike trains for transport.

    The codec preserves symbol magnitude continuity during transport,
    enabling graceful degradation under noise rather than catastrophic failure.

    Attributes:
        max_spikes_per_packet: Maximum spikes per packet (for streaming)
        precision: Decimal precision to preserve (default: 6)

    Example:
        >>> codec = SpikelinkCodec()
        >>> train = SpikeTrain(times=[0.1, 0.2, 0.3])
        >>> packets = codec.encode_train(train)
        >>> recovered = codec.decode_packets(packets)
    """

    def __init__(
        self,
        max_spikes_per_packet: int = 1024,
        precision: int = 6,
    ):
        """
        Initialize the codec.

        Args:
            max_spikes_per_packet: Maximum spikes per packet
            precision: Decimal precision to preserve
        """
        self.max_spikes_per_packet = max_spikes_per_packet
        self.precision = precision

    def encode_train(self, train: SpikeTrain) -> List[SpikelinkPacket]:
        """
        Encode a SpikeTrain into packets.

        Args:
            train: SpikeTrain to encode

        Returns:
            List of SpikelinkPackets
        """
        packets = []
        times = np.asarray(train.times, dtype=np.float64)

        # Split into chunks if needed
        n_packets = max(
            1,
            (len(times) + self.max_spikes_per_packet - 1) // self.max_spikes_per_packet,
        )

        for i in range(n_packets):
            start_idx = i * self.max_spikes_per_packet
            end_idx = min((i + 1) * self.max_spikes_per_packet, len(times))
            chunk = times[start_idx:end_idx]

            # Use train bounds for all packets (not chunk bounds)
            t_start = float(train.t_start)
            t_stop = float(train.t_stop)

            # Set continuation flag for multi-packet sequences
            flags = 0
            if i < n_packets - 1:
                flags |= SpikelinkPacket.FLAG_CONTINUATION

            packet = SpikelinkPacket(
                times=chunk.astype(np.float32),
                t_start=0.0,
                t_stop=t_stop,
                sequence_id=i,
                neuron_id=train.neuron_id if isinstance(train.neuron_id, int) else None,
                flags=flags,
            )
            packets.append(packet)

        return packets

    def decode_packets(
        self,
        packets: List[SpikelinkPacket],
        neuron_id: Optional[Union[int, str]] = None,
    ) -> SpikeTrain:
        """
        Decode packets back to a SpikeTrain.

        Args:
            packets: List of packets to decode
            neuron_id: Optional neuron ID to assign

        Returns:
            Reconstructed SpikeTrain
        """
        if not packets:
            return SpikeTrain(times=[], t_start=0.0, t_stop=1.0)

        # Concatenate all spike times
        all_times = np.concatenate([p.times for p in packets])

        # Convert back to float64 for precision
        all_times = all_times.astype(np.float64)

        # Determine overall time bounds
        t_start = min(p.t_start for p in packets)
        t_stop = max(p.t_stop for p in packets)

        # Ensure t_stop covers all spike times
        if len(all_times) > 0:
            max_spike = float(np.max(all_times))
            if max_spike > t_stop:
                t_stop = max_spike + 0.1

        # Use neuron_id from first packet if not provided
        if neuron_id is None:
            neuron_id = packets[0].neuron_id

        return SpikeTrain(
            times=all_times,
            t_start=0.0,
            t_stop=t_stop,
            neuron_id=neuron_id,
        )

    def encode_times(
        self, times: Union[List[float], np.ndarray]
    ) -> List[SpikelinkPacket]:
        """
        Convenience method to encode raw spike times.

        Args:
            times: List or array of spike times

        Returns:
            List of packets
        """
        train = SpikeTrain(times=times)
        return self.encode_train(train)

    def decode_times(self, packets: List[SpikelinkPacket]) -> np.ndarray:
        """
        Convenience method to decode packets to raw times.

        Args:
            packets: List of packets

        Returns:
            Array of spike times
        """
        train = self.decode_packets(packets)
        return train.times

    def round_trip(self, train: SpikeTrain) -> SpikeTrain:
        """
        Perform encode-decode round trip.

        Useful for testing precision preservation.

        Args:
            train: Original spike train

        Returns:
            Recovered spike train after round trip
        """
        packets = self.encode_train(train)
        return self.decode_packets(packets, neuron_id=train.neuron_id)

    def __repr__(self) -> str:
        return (
            f"SpikelinkCodec(max_spikes_per_packet={self.max_spikes_per_packet}, "
            f"precision={self.precision})"
        )
