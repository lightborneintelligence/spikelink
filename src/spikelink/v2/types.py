"""
SpikeLink v2 Types — Wave-enhanced spike representations.
Lightborne Intelligence
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class V2SpikeTrain:
    """A sequence of spike events with timing and amplitude.
    
    Unlike v1 SpikeTrain (times only), v2 carries amplitude information
    which enables wave-domain processing via Harmonic Transform.
    
    Attributes:
        times: Spike timestamps in seconds (float64)
        amplitudes: Spike amplitudes as continuous values (float64)
    """
    times: np.ndarray      # Spike timestamps (seconds)
    amplitudes: np.ndarray  # Spike amplitudes (continuous)
    
    def __post_init__(self):
        """Ensure arrays are numpy float64."""
        if not isinstance(self.times, np.ndarray):
            self.times = np.array(self.times, dtype=np.float64)
        if not isinstance(self.amplitudes, np.ndarray):
            self.amplitudes = np.array(self.amplitudes, dtype=np.float64)
    
    @property
    def count(self) -> int:
        """Number of spikes in this train."""
        return len(self.times)
    
    @property
    def duration(self) -> float:
        """Duration from first to last spike (seconds)."""
        if self.count == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])
    
    @property
    def mean_rate(self) -> float:
        """Mean firing rate (Hz)."""
        if self.duration == 0:
            return 0.0
        return self.count / self.duration
    
    def copy(self) -> 'V2SpikeTrain':
        """Return a deep copy."""
        return V2SpikeTrain(
            times=self.times.copy(),
            amplitudes=self.amplitudes.copy()
        )
    
    def __repr__(self) -> str:
        return (f"V2SpikeTrain(count={self.count}, "
                f"duration={self.duration:.3f}s, "
                f"rate={self.mean_rate:.1f}Hz)")


# Alias for backward compatibility within v2 module
SpikeTrain = V2SpikeTrain

# Public API alias (keeps top-level import stable)
SpikeTrainV2 = V2SpikeTrain
