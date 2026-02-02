"""
SpikeLink v2 Packet — Wave-enhanced wire format.
Lightborne Intelligence

40-byte packet carrying wave-decomposed spike data with:
- 7-shell harmonic coefficients (amplitude + phase)
- ERA metadata for identity protection
- Absolute timing anchor (10μs resolution)
- Phase coherence tracking
"""

import struct
from dataclasses import dataclass, field
from typing import List


# Protocol constants
MAGIC_V2 = b'SPK2'           # Distinguishes v2 from v1 'SPKS'
PACKET_SIZE_V2 = 40          # 40 bytes (v1 was 32)
PROTOCOL_VERSION = 0x20      # v2.0


@dataclass
class SpikelinkPacketV2:
    """
    SpikeLink v2 Packet — Wave-Enhanced.
    
    Layout (40 bytes):
    ┌────────┬────────┬───────┬───────┬──────────────┬──────────┬──────────┐
    │ Magic  │Version │ Count │ Cycle │  Shell Data   │ ERA Meta │ Coherence│
    │4 bytes │1 byte  │1 byte │2 bytes│  24 bytes     │ 4 bytes  │ 4 bytes  │
    └────────┴────────┴───────┴───────┴──────────────┴──────────┴──────────┘
    
    Shell Data (24 bytes = 7 shells × 3 bytes + 3 reserved):
      Each shell: [amplitude_hi][amplitude_lo][phase_byte]
      - amplitude: uint16 with tier-adaptive precision
      - phase: uint8 mapping [0,255] → [-π, π]
    
    ERA Meta (4 bytes):
      [tier_map_lo][era_correction_flags][energy_byte][tier_map_hi]
      - tier_map_lo: shells 0-3 tier assignment (2 bits each)
      - tier_map_hi: shells 4-6 tier assignment (2 bits each)
      - Full 14-bit tier map across both bytes
    
    Coherence (4 bytes):
      [phase_ref_hi][phase_ref_lo][sequence_hi][sequence_lo]
      - phase_ref: inter-packet phase reference for continuity
      - sequence: packet sequence number for drift detection
    """
    # Header
    magic: bytes = MAGIC_V2
    version: int = PROTOCOL_VERSION
    shell_count: int = 7
    cycle_us: int = 0
    
    # Shell data (wave-native)
    shell_amplitudes: List[int] = field(default_factory=lambda: [0] * 7)
    shell_phases: List[int] = field(default_factory=lambda: [0] * 7)
    
    # ERA metadata
    shell_tier_map: int = 0        # Packed tier assignment: shells 0-3 (2 bits each)
    era_correction_flags: int = 0  # Which shells were ERA-corrected
    energy_byte: int = 0           # Coefficient scale quantized to uint8
    shell_tier_map_hi: int = 0     # Packed tier assignment: shells 4-6 (2 bits each)
    
    # Coherence tracking
    phase_reference: int = 0       # uint16 inter-packet phase ref
    sequence_number: int = 0       # uint16 packet sequence
    
    # Timing (encoded in reserved bytes)
    chunk_start_10us: int = 0      # uint24 — absolute start time in 10μs units
    
    @property
    def spike_count(self) -> int:
        """Actual number of spikes in this packet.
        
        Wire field is named 'shell_count' for backward compatibility,
        but it carries the real spike count (not number of HT shells).
        """
        return self.shell_count
    
    @spike_count.setter
    def spike_count(self, value: int):
        self.shell_count = value
    
    def pack(self) -> bytes:
        """Serialize to 40-byte wire format."""
        buf = bytearray(PACKET_SIZE_V2)
        
        # Magic (4 bytes)
        buf[0:4] = self.magic
        
        # Version (1 byte)
        buf[4] = self.version
        
        # Shell count (1 byte) — actual spike count for this packet
        buf[5] = self.shell_count
        
        # Cycle (2 bytes, little-endian)
        struct.pack_into('<H', buf, 6, self.cycle_us)
        
        # Shell data: 7 × (2 bytes amplitude + 1 byte phase) = 21 bytes
        for i in range(7):
            offset = 8 + i * 3
            struct.pack_into('<H', buf, offset, 
                            self.shell_amplitudes[i] & 0xFFFF)
            buf[offset + 2] = self.shell_phases[i] & 0xFF
        
        # Reserved bytes 29-31: chunk_start_10us (24-bit unsigned)
        val = self.chunk_start_10us & 0xFFFFFF
        buf[29] = val & 0xFF
        buf[30] = (val >> 8) & 0xFF
        buf[31] = (val >> 16) & 0xFF
        
        # ERA Meta (4 bytes at offset 32)
        buf[32] = self.shell_tier_map & 0xFF        # shells 0-3
        buf[33] = self.era_correction_flags & 0xFF
        buf[34] = self.energy_byte & 0xFF
        buf[35] = self.shell_tier_map_hi & 0xFF     # shells 4-6
        
        # Coherence (4 bytes at offset 36)
        struct.pack_into('<H', buf, 36, self.phase_reference & 0xFFFF)
        struct.pack_into('<H', buf, 38, self.sequence_number & 0xFFFF)
        
        return bytes(buf)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'SpikelinkPacketV2':
        """Deserialize from 40-byte wire format."""
        if len(data) < PACKET_SIZE_V2:
            raise ValueError(f"Packet too short: {len(data)} < {PACKET_SIZE_V2}")
        
        pkt = cls()
        pkt.magic = data[0:4]
        
        if pkt.magic != MAGIC_V2:
            raise ValueError(f"Invalid magic: {pkt.magic}")
        
        pkt.version = data[4]
        pkt.shell_count = data[5]
        pkt.cycle_us = struct.unpack_from('<H', data, 6)[0]
        
        # Shell data
        pkt.shell_amplitudes = []
        pkt.shell_phases = []
        for i in range(7):
            offset = 8 + i * 3
            amp = struct.unpack_from('<H', data, offset)[0]
            ph = data[offset + 2]
            pkt.shell_amplitudes.append(amp)
            pkt.shell_phases.append(ph)
        
        # ERA Meta
        pkt.shell_tier_map = data[32]           # shells 0-3
        pkt.era_correction_flags = data[33]
        pkt.energy_byte = data[34]
        pkt.shell_tier_map_hi = data[35]        # shells 4-6
        
        # Chunk start time from reserved bytes 29-31
        pkt.chunk_start_10us = data[29] | (data[30] << 8) | (data[31] << 16)
        
        # Coherence
        pkt.phase_reference = struct.unpack_from('<H', data, 36)[0]
        pkt.sequence_number = struct.unpack_from('<H', data, 38)[0]
        
        return pkt
    
    def __repr__(self) -> str:
        return (f"SpikelinkPacketV2(spikes={self.spike_count}, "
                f"seq={self.sequence_number}, "
                f"t_start={self.chunk_start_10us * 1e-5:.4f}s)")
