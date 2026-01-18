"""
SpikelinkPacket — Wire format for spike transport.

Packets encode spike times as symbol magnitudes with bounded precision,
enabling graceful degradation under noise rather than catastrophic failure.
"""

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Packet format constants
MAGIC = b"SPLK"  # 4 bytes: SpikeLink magic number
VERSION = 1  # Protocol version
HEADER_SIZE = 16  # bytes


@dataclass
class SpikelinkPacket:
    """
    A transport packet containing encoded spike data.

    The packet format is designed for:
    - Symbol magnitude preservation (not binary quantization)
    - Bounded precision with explicit degradation characteristics
    - Streaming-friendly fixed headers
    - Optional integrity verification

    Wire format:
        [MAGIC:4][VERSION:1][FLAGS:1][COUNT:2][T_START:4][T_STOP:4][PAYLOAD:N*4]

    Attributes:
        times: Encoded spike times (float32 for transport)
        t_start: Time window start
        t_stop: Time window stop
        sequence_id: Optional packet sequence number
        neuron_id: Optional source neuron identifier
        flags: Packet flags (compression, integrity, etc.)
    """

    times: np.ndarray
    t_start: float = 0.0
    t_stop: float = 1.0
    sequence_id: Optional[int] = None
    neuron_id: Optional[int] = None
    flags: int = 0

    # Flag definitions
    FLAG_COMPRESSED = 0x01
    FLAG_HAS_CHECKSUM = 0x02
    FLAG_CONTINUATION = 0x04

    def __post_init__(self):
        """Ensure times are float32 for transport."""
        if not isinstance(self.times, np.ndarray):
            self.times = np.array(self.times, dtype=np.float32)
        elif self.times.dtype != np.float32:
            self.times = self.times.astype(np.float32)

    def __len__(self) -> int:
        """Return number of spikes in packet."""
        return len(self.times)

    @property
    def payload_size(self) -> int:
        """Return payload size in bytes."""
        return len(self.times) * 4  # float32 = 4 bytes

    @property
    def total_size(self) -> int:
        """Return total packet size in bytes."""
        return HEADER_SIZE + self.payload_size

    def to_bytes(self) -> bytes:
        """
        Serialize packet to bytes.

        Returns:
            Bytes representation of the packet
        """
        # Header: MAGIC(4) + VERSION(1) + FLAGS(1) + COUNT(2) + T_START(4) + T_STOP(4)
        header = struct.pack(
            ">4sBBHff",
            MAGIC,
            VERSION,
            self.flags,
            len(self.times),
            self.t_start,
            self.t_stop,
        )

        # Payload: spike times as float32
        payload = self.times.tobytes()

        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "SpikelinkPacket":
        """
        Deserialize packet from bytes.

        Args:
            data: Raw bytes containing packet data

        Returns:
            Decoded SpikelinkPacket

        Raises:
            ValueError: If magic number doesn't match or data is corrupted
        """
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Packet too short: {len(data)} < {HEADER_SIZE}")

        # Parse header
        magic, version, flags, count, t_start, t_stop = struct.unpack(
            ">4sBBHff", data[:HEADER_SIZE]
        )

        if magic != MAGIC:
            raise ValueError(f"Invalid magic number: {magic}")

        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        # Parse payload
        expected_size = HEADER_SIZE + count * 4
        if len(data) < expected_size:
            raise ValueError(f"Packet truncated: {len(data)} < {expected_size}")

        times = np.frombuffer(
            data[HEADER_SIZE : HEADER_SIZE + count * 4], dtype=np.float32
        )

        return cls(
            times=times.copy(),
            t_start=t_start,
            t_stop=t_stop,
            flags=flags,
        )

    def verify_integrity(self) -> bool:
        """
        Verify packet integrity.

        Returns:
            True if packet is valid, False otherwise
        """
        # Check time bounds
        if len(self.times) > 0:
            if np.any(self.times < self.t_start) or np.any(self.times > self.t_stop):
                return False

        # Check for NaN/Inf
        if np.any(~np.isfinite(self.times)):
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"SpikelinkPacket(n_spikes={len(self)}, "
            f"t_start={self.t_start:.3f}, t_stop={self.t_stop:.3f}, "
            f"size={self.total_size} bytes)"
        )
