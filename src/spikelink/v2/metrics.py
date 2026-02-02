"""
SpikeLink v2 Metrics — Transport quality measurement.
Lightborne Intelligence

Quantitative comparison framework for v1 vs v2 transports.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from spikelink.waveml import WaveState
from spikelink.v2.types import V2SpikeTrain
from spikelink.v2.packet import PACKET_SIZE_V2


@dataclass
class TransportMetrics:
    """Quantitative comparison between v1 and v2 transports.
    
    Attributes:
        reconstruction_error: MSE between original and recovered amplitudes
        identity_preservation: Shell 0-1 energy ratio (recovered/original)
        noise_suppression: Shell 6 energy ratio (recovered/original)
        phase_coherence: Mean phase error across packets
        energy_ratio: Total energy ratio (recovered/original)
        packets_used: Number of packets used for transport
        bytes_per_spike: Transport overhead metric
    """
    reconstruction_error: float
    identity_preservation: float
    noise_suppression: float
    phase_coherence: float
    energy_ratio: float
    packets_used: int
    bytes_per_spike: float
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"TransportMetrics:\n"
            f"  Reconstruction error: {self.reconstruction_error:.6f}\n"
            f"  Identity preservation: {self.identity_preservation:.2%}\n"
            f"  Noise suppression: {self.noise_suppression:.2%}\n"
            f"  Phase coherence error: {self.phase_coherence:.6f}\n"
            f"  Energy ratio: {self.energy_ratio:.2%}\n"
            f"  Packets used: {self.packets_used}\n"
            f"  Bytes per spike: {self.bytes_per_spike:.2f}"
        )


def compute_metrics(
    original: V2SpikeTrain, 
    recovered: V2SpikeTrain,
    original_state: Optional[WaveState] = None,
    recovered_state: Optional[WaveState] = None,
    spikes_per_packet: int = 7
) -> TransportMetrics:
    """Compute transport quality metrics.
    
    Args:
        original: Original spike train before transport
        recovered: Recovered spike train after transport
        original_state: Optional WaveState for shell-level analysis
        recovered_state: Optional WaveState for shell-level analysis
        spikes_per_packet: Chunk size used during encoding. Must match
            the codec's spikes_per_packet to produce correct packet count
            and bytes-per-spike figures.
    
    Returns:
        TransportMetrics with all quality measurements
    """
    
    # Reconstruction error (amplitude MSE)
    min_len = min(original.count, recovered.count)
    if min_len == 0:
        mse = float('inf')
    else:
        mse = float(np.mean(
            (original.amplitudes[:min_len] - recovered.amplitudes[:min_len]) ** 2))
    
    # Shell-level metrics (if wave states provided)
    id_pres = 1.0
    noise_sup = 1.0
    if original_state is not None and recovered_state is not None:
        # Identity preservation: how well shells 0-1 survived
        orig_id_energy = float(np.sum(original_state.amplitude[:2] ** 2))
        recv_id_energy = float(np.sum(recovered_state.amplitude[:2] ** 2))
        id_pres = recv_id_energy / (orig_id_energy + 1e-12)
        
        # Noise suppression: how much shell 6 was reduced
        orig_noise = float(original_state.amplitude[-1] ** 2)
        recv_noise = float(recovered_state.amplitude[-1] ** 2)
        noise_sup = recv_noise / (orig_noise + 1e-12)
    
    # Phase coherence
    if min_len > 1:
        orig_phases = np.diff(original.times[:min_len])
        recv_phases = np.diff(recovered.times[:min_len])
        min_ph = min(len(orig_phases), len(recv_phases))
        phase_err = float(np.mean(np.abs(
            orig_phases[:min_ph] - recv_phases[:min_ph])))
    else:
        phase_err = 0.0
    
    # Energy ratio
    orig_energy = float(np.sum(original.amplitudes[:min_len] ** 2))
    recv_energy = float(np.sum(recovered.amplitudes[:min_len] ** 2))
    energy_ratio = recv_energy / (orig_energy + 1e-12)
    
    n_packets = int(np.ceil(original.count / spikes_per_packet))
    bytes_per_spike = (n_packets * PACKET_SIZE_V2) / (original.count + 1e-12)
    
    return TransportMetrics(
        reconstruction_error=mse,
        identity_preservation=id_pres,
        noise_suppression=noise_sup,
        phase_coherence=phase_err,
        energy_ratio=energy_ratio,
        packets_used=n_packets,
        bytes_per_spike=bytes_per_spike
    )
