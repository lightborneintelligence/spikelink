"""
SpikeLink v2 Codec — Wave-enhanced encoding/decoding.
Lightborne Intelligence

The codec transforms spike trains through:
  Encode: SpikeTrain → HT → ERA → PrecisionAlloc → Pack → Packets
  Decode: Packets → Unpack → PrecisionDecode → ERA → HT Inverse → SpikeTrain

The fundamental upgrade from v1:
  v1 transports spike amplitudes.
  v2 transports wave-decomposed identity.
"""

import numpy as np
from typing import List, Optional, Tuple
from math import pi as PI

# Package imports
from spikelink.waveml import (
    WaveState, HarmonicTransform, ERA, ERABounds,
    ShellMap, ShellTier
)
from spikelink.v2.packet import SpikelinkPacketV2, PACKET_SIZE_V2
from spikelink.v2.types import V2SpikeTrain


class PrecisionAllocator:
    """
    Allocates encoding precision across shells based on tier.
    
    Identity shells get full uint16 range (16 bits effective).
    Noise shells get reduced range (fewer effective bits).
    
    This is WHERE wave-awareness enters the transport:
    The protocol *knows* which shells matter and gives them
    more bits.
    """
    
    def __init__(self, shell_map: ShellMap):
        self.shell_map = shell_map
        self.weights = shell_map.precision_weights()
        
        # Effective bit allocation per tier
        self.tier_bits = {
            ShellTier.IDENTITY:  16,  # Full uint16
            ShellTier.STRUCTURE: 14,  # 14-bit effective
            ShellTier.DYNAMICS:  12,  # 12-bit effective
            ShellTier.NOISE:     8,   # 8-bit effective
        }
    
    def encode_amplitude(self, shell_idx: int, amplitude: float, 
                         max_amplitude: float) -> int:
        """
        Encode amplitude to uint16 with tier-adaptive precision.
        
        Identity shells: full 16-bit resolution
        Noise shells: quantized to fewer effective bits
        """
        tier = self.shell_map.tier(shell_idx)
        effective_bits = self.tier_bits[tier]
        
        # Normalize to [0, 1]
        normalized = min(abs(amplitude) / (max_amplitude + 1e-12), 1.0)
        
        # Quantize to effective bit depth, then scale to uint16
        max_val = (1 << effective_bits) - 1
        quantized = int(normalized * max_val)
        
        # Scale quantized value to full uint16 range
        if effective_bits < 16:
            scale = 65535 / max_val
            return min(int(quantized * scale), 65535)
        else:
            return min(quantized, 65535)
    
    def decode_amplitude(self, shell_idx: int, encoded: int,
                         max_amplitude: float) -> float:
        """
        Decode uint16 back to amplitude with tier-aware dequantization.
        """
        tier = self.shell_map.tier(shell_idx)
        effective_bits = self.tier_bits[tier]
        
        max_val = (1 << effective_bits) - 1
        
        # Reverse the scale
        if effective_bits < 16:
            scale = 65535 / max_val
            quantized = int(encoded / scale + 0.5)
        else:
            quantized = encoded
        
        # Dequantize
        normalized = quantized / max_val
        return normalized * max_amplitude
    
    @staticmethod
    def encode_phase(phase: float) -> int:
        """Map phase [-π, π] → uint8 [0, 255]."""
        normalized = (phase + PI) / (2 * PI)
        return int(np.clip(normalized * 255, 0, 255))
    
    @staticmethod
    def decode_phase(encoded: int) -> float:
        """Map uint8 [0, 255] → phase [-π, π]."""
        normalized = encoded / 255.0
        return normalized * 2 * PI - PI


class SpikelinkCodecV2:
    """
    SpikeLink v2 Wave-Enhanced Codec.
    
    Encoding pipeline:
      1. Chunk spike train into segments (7 spikes per packet)
      2. HT forward: spike segment → WaveState (shell decomposition)
      3. ERA pre-encode: protect identity shells before compression
      4. Precision allocate: tier-adaptive amplitude encoding
      5. Phase coherence: track inter-packet phase reference
      6. Pack: serialize to 40-byte wire packet
    
    Decoding pipeline:
      1. Unpack: deserialize from wire
      2. Precision decode: recover amplitude per tier
      3. ERA post-decode: correct transport-induced drift
      4. HT inverse: WaveState → spike segment
      5. Phase coherence: reconstruct timing continuity
      6. Assemble: concatenate segments into full spike train
    
    The key difference from v1:
      v1 transports spike amplitudes.
      v2 transports wave-decomposed identity.
    """
    
    def __init__(self,
                 spikes_per_packet: int = 7,
                 shell_map: Optional[ShellMap] = None,
                 era_bounds: Optional[ERABounds] = None,
                 curvature_scale: float = 1.1,
                 max_amplitude: float = 10.0):
        
        self.spikes_per_packet = spikes_per_packet
        self.shell_map = shell_map or ShellMap(n_shells=7)
        self.ht = HarmonicTransform(n_shells=7, curvature_scale=curvature_scale)
        self.era = ERA(self.shell_map, era_bounds or ERABounds())
        self.allocator = PrecisionAllocator(self.shell_map)
        # Pre-HT amplitude clamp: prevents absurd values from
        # blowing up the coefficient scale and wasting precision.
        # Without this, a single spike at amplitude 1e6 would
        # compress all other amplitudes to near-zero in the uint16 range.
        self.max_amplitude = max_amplitude
        
        # State tracking
        self._sequence = 0
        self._prev_state: Optional[WaveState] = None
        self._prev_prev_state: Optional[WaveState] = None
        self._phase_ref = 0.0
    
    def _chunk_train(self, train: V2SpikeTrain) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Split spike train into chunks of spikes_per_packet.
        Returns list of (times, amps, actual_count) tuples."""
        chunks = []
        n = train.count
        
        for start in range(0, n, self.spikes_per_packet):
            end = min(start + self.spikes_per_packet, n)
            times = train.times[start:end]
            amps = train.amplitudes[start:end]
            actual_count = len(times)
            
            # Pad if needed
            if len(times) < self.spikes_per_packet:
                pad_count = self.spikes_per_packet - len(times)
                times = np.concatenate([times, np.zeros(pad_count)])
                amps = np.concatenate([amps, np.zeros(pad_count)])
            
            chunks.append((times, amps, actual_count))
        
        return chunks
    
    def _pack_tier_map(self) -> Tuple[int, int]:
        """Pack all 7 shell tier assignments into 14 bits across 2 bytes.
        
        Each shell gets 2 bits (0=IDENTITY, 1=STRUCTURE, 2=DYNAMICS, 3=NOISE).
        7 shells × 2 bits = 14 bits.
        
        Returns:
            (lo_byte, hi_byte): shells 0-3 in lo_byte, shells 4-6 in hi_byte.
        """
        packed = 0
        for i in range(min(7, self.shell_map.n_shells)):
            tier = self.shell_map.tier(i)
            packed |= (int(tier) & 0x03) << (i * 2)
        lo = packed & 0xFF          # shells 0-3 (bits 0-7)
        hi = (packed >> 8) & 0xFF   # shells 4-6 (bits 8-13)
        return (lo, hi)
    
    def encode_train(self, train: V2SpikeTrain) -> List[SpikelinkPacketV2]:
        """
        Full v2 encoding pipeline:
        SpikeTrain → [HT → ERA → PrecisionAlloc → Pack] → Packets
        
        Key design: the max coefficient amplitude is computed per-packet
        and stored in the energy_byte field for decode-side rescaling.
        This ensures the full uint16 range maps to actual coefficient range,
        not a fixed max_amplitude that may waste precision.
        """
        if train.count == 0:
            return []
        
        chunks = self._chunk_train(train)
        packets = []
        self._coeff_scales = []  # Track per-packet scale for decode
        
        for chunk_times, chunk_amps, actual_count in chunks:
            # ── Step 0: Pre-HT Amplitude Clamp ──
            # Prevent outlier amplitudes from dominating the coefficient scale.
            # Clamps to [-max_amplitude, +max_amplitude] before decomposition.
            chunk_amps = np.clip(chunk_amps, -self.max_amplitude, self.max_amplitude)
            
            # ── Step 1: Harmonic Transform ──
            # Decompose spike amplitudes into wave shells
            wave_state = self.ht.forward(chunk_amps)
            
            # ── Step 2: ERA Pre-Encode ──
            # Protect identity before transport compression
            if self._prev_prev_state is not None:
                wave_state = self.era.adaptive_rectify(
                    wave_state, self._prev_state, self._prev_prev_state)
            elif self._prev_state is not None:
                wave_state = self.era.rectify(wave_state, self._prev_state)
            else:
                wave_state = self.era.rectify(wave_state)
            
            # ── Step 3: Compute per-packet coefficient scale ──
            # Use the actual max coefficient as the encoding range
            coeff_max = float(np.max(wave_state.amplitude)) + 1e-12
            self._coeff_scales.append(coeff_max)
            
            # ── Step 4: Precision Allocation ──
            # Tier-adaptive amplitude encoding using actual coefficient range
            pkt = SpikelinkPacketV2()
            pkt.shell_count = actual_count  # ACTUAL spike count, not always 7
            
            enc_amps = []
            enc_phases = []
            era_flags = 0
            
            for i in range(7):
                enc_amp = self.allocator.encode_amplitude(
                    i, wave_state.amplitude[i], coeff_max)
                enc_ph = self.allocator.encode_phase(wave_state.phase[i])
                enc_amps.append(enc_amp)
                enc_phases.append(enc_ph)
                
                # Track if ERA modified this shell
                if self._prev_state is not None:
                    if abs(wave_state.amplitude[i] - 
                           self._prev_state.amplitude[i]) > 0.01:
                        era_flags |= (1 << i)
            
            pkt.shell_amplitudes = enc_amps
            pkt.shell_phases = enc_phases
            
            # ── Step 5: Timing ──
            t_base = float(chunk_times[0]) if actual_count > 0 else 0.0
            t_span = float(chunk_times[actual_count - 1] - chunk_times[0]) if actual_count > 1 else 0.0
            pkt.cycle_us = min(int(t_span * 1e6), 65535)
            # Store absolute chunk start time (10μs resolution, covers 167s)
            pkt.chunk_start_10us = min(int(t_base * 1e5), 0xFFFFFF)
            
            # ── Step 6: ERA Metadata ──
            tier_lo, tier_hi = self._pack_tier_map()
            pkt.shell_tier_map = tier_lo
            pkt.era_correction_flags = era_flags & 0xFF
            # Encode coeff_max into energy_byte (log scale for wider range)
            # Range: 0.01 → 100.0 mapped to 0 → 255
            log_scale = np.log10(max(coeff_max, 0.01)) + 2  # shift so 0.01→0
            pkt.energy_byte = min(int(log_scale * 63.75), 255)  # 4.0 * 63.75 = 255
            pkt.shell_tier_map_hi = tier_hi  # shells 4-6 in reserved ERA byte
            
            # ── Step 7: Phase Coherence ──
            if self._prev_state is not None:
                phase_delta = float(
                    np.mean(wave_state.phase[:2] - self._prev_state.phase[:2]))
                self._phase_ref += phase_delta
            
            pkt.phase_reference = int(
                ((self._phase_ref + PI) / (2 * PI)) * 65535) & 0xFFFF
            pkt.sequence_number = self._sequence & 0xFFFF
            
            # ── Update state tracking ──
            self._prev_prev_state = (self._prev_state.copy() 
                                      if self._prev_state else None)
            self._prev_state = wave_state.copy()
            self._sequence += 1
            
            packets.append(pkt)
        
        return packets
    
    def decode_packets(self, packets: List[SpikelinkPacketV2]) -> V2SpikeTrain:
        """
        Full v2 decoding pipeline:
        Packets → [Unpack → PrecisionDecode → ERA PostDecode → HT Inverse] → SpikeTrain
        """
        if not packets:
            return V2SpikeTrain(times=np.array([]), amplitudes=np.array([]))
        
        all_times = []
        all_amps = []
        
        decode_era = ERA(self.shell_map, ERABounds(
            # Transport-decode bounds: relaxed vs signal-domain
            # We just need to catch catastrophic drift, not shape identity
            identity_max_amplitude=5.0,
            identity_max_energy=25.0,
            identity_max_phase_drift=0.5,
            structure_max_amplitude=4.0,
            structure_max_energy=20.0,
            structure_max_phase_drift=0.8,
            dynamics_max_amplitude=3.0,
            dynamics_max_energy=15.0,
            dynamics_max_phase_drift=1.5,
            noise_max_amplitude=1.5,
            noise_max_energy=5.0,
            noise_max_phase_drift=PI,
        ))
        prev_state = None
        
        for pkt in packets:
            # ── Step 1: Recover coefficient scale from energy_byte ──
            log_scale = pkt.energy_byte / 63.75
            coeff_max = 10.0 ** (log_scale - 2)  # Reverse the log mapping
            
            # ── Step 2: Precision Decode with correct scale ──
            amp_decoded = np.zeros(7)
            phase_decoded = np.zeros(7)
            
            for i in range(7):
                amp_decoded[i] = self.allocator.decode_amplitude(
                    i, pkt.shell_amplitudes[i], coeff_max)
                phase_decoded[i] = self.allocator.decode_phase(
                    pkt.shell_phases[i])
            
            wave_state = WaveState(amplitude=amp_decoded, phase=phase_decoded)
            
            # ── Step 3: ERA Post-Decode ──
            wave_state = decode_era.rectify(wave_state, prev_state)
            prev_state = wave_state.copy()
            
            # ── Step 4: HT Inverse ──
            reconstructed = self.ht.inverse(wave_state, self.spikes_per_packet)
            
            # ── Step 5: Timing Reconstruction ──
            actual_count = min(pkt.shell_count, self.spikes_per_packet)
            if actual_count == 0:
                actual_count = self.spikes_per_packet  # fallback
            
            # Absolute start time from packet (10μs resolution)
            t_start = pkt.chunk_start_10us * 1e-5
            t_span = pkt.cycle_us / 1e6
            
            if t_span > 0 and actual_count > 1:
                times = t_start + np.linspace(0, t_span, actual_count)
            elif actual_count == 1:
                times = np.array([t_start])
            else:
                times = np.full(actual_count, t_start)
            
            # Take only the first actual_count reconstructed amplitudes
            amps_out = reconstructed[:actual_count]
            
            all_times.extend(times.tolist())
            all_amps.extend(amps_out.tolist())
        
        return V2SpikeTrain(
            times=np.array(all_times),
            amplitudes=np.array(all_amps)
        )
    
    def reset(self):
        """Reset codec state for new stream."""
        self._sequence = 0
        self._prev_state = None
        self._prev_prev_state = None
        self._phase_ref = 0.0
        self.era.reset()
    
    def __repr__(self) -> str:
        return (f"SpikelinkCodecV2(spikes_per_packet={self.spikes_per_packet}, "
                f"max_amplitude={self.max_amplitude})")
