"""
Neo / EBRAINS adapter for SpikeLink.

Provides bidirectional conversion between Neo SpikeTrain and SpikeLink formats.
Requires: pip install spikelink[neo]
"""

import numpy as np
from ..types.spiketrain import SpikeTrain
from ..v2.types import V2SpikeTrain


class NeoAdapter:
    """
    Bridge between Neo SpikeTrain and SpikeLink v1 SpikeTrain.
    """
    
    @staticmethod
    def _check_neo():
        try:
            import neo
            import quantities as pq
            return neo, pq
        except ImportError:
            raise ImportError(
                "NeoAdapter requires 'neo' and 'quantities' packages. "
                "Install with: pip install spikelink[neo]"
            )
    
    @staticmethod
    def from_neo(neo_train) -> SpikeTrain:
        """Convert Neo SpikeTrain → SpikeLink v1 SpikeTrain."""
        neo, pq = NeoAdapter._check_neo()
        times = np.array(neo_train.rescale('s').magnitude, dtype=np.float64)
        t_start = float(neo_train.t_start.rescale('s').magnitude)
        t_stop = float(neo_train.t_stop.rescale('s').magnitude)
        return SpikeTrain(times=times, t_start=t_start, t_stop=t_stop)
    
    @staticmethod
    def to_neo(spiketrain: SpikeTrain, t_stop: float = None):
        """Convert SpikeLink v1 SpikeTrain → Neo SpikeTrain."""
        neo, pq = NeoAdapter._check_neo()
        times = spiketrain.times
        if t_stop is None:
            t_stop = spiketrain.t_stop if hasattr(spiketrain, 't_stop') else (
                float(times[-1]) + 0.1 if len(times) > 0 else 1.0
            )
        return neo.SpikeTrain(times=times * pq.s, t_stop=t_stop * pq.s)
    
    @staticmethod
    def is_available() -> bool:
        """Check if Neo package is available."""
        try:
            import neo
            import quantities
            return True
        except ImportError:
            return False


class NeoAdapterV2:
    """
    Bridge between Neo SpikeTrain and SpikeLink v2 SpikeTrain.
    
    Neo SpikeTrain: times with units, no amplitudes
    v2 SpikeTrain:  times + amplitudes (wave-enhanced)
    
    Strategy:
      Neo → v2: assign unit amplitudes (timing IS the data)
      v2 → Neo: drop amplitudes, preserve times with units
    """
    
    @staticmethod
    def _check_neo():
        try:
            import neo
            import quantities as pq
            return neo, pq
        except ImportError:
            raise ImportError(
                "NeoAdapterV2 requires 'neo' and 'quantities' packages. "
                "Install with: pip install spikelink[neo]"
            )
    
    @staticmethod
    def from_neo(neo_train, default_amplitude: float = 1.0) -> V2SpikeTrain:
        """Convert Neo SpikeTrain → v2 SpikeTrain."""
        neo, pq = NeoAdapterV2._check_neo()
        times = np.array(neo_train.rescale('s').magnitude, dtype=np.float64)
        amplitudes = np.full(len(times), default_amplitude)
        return V2SpikeTrain(times=times, amplitudes=amplitudes)
    
    @staticmethod
    def to_neo(v2_train: V2SpikeTrain, t_stop: float = None):
        """Convert v2 SpikeTrain → Neo SpikeTrain."""
        neo, pq = NeoAdapterV2._check_neo()
        times = v2_train.times
        if t_stop is None:
            t_stop = float(times[-1]) + 0.1 if len(times) > 0 else 1.0
        return neo.SpikeTrain(times=times * pq.s, t_stop=t_stop * pq.s)
    
    @staticmethod
    def from_v1_spikelink(v1_train) -> V2SpikeTrain:
        """Convert published spikelink v1 SpikeTrain → v2 SpikeTrain."""
        times = np.array(v1_train.times, dtype=np.float64)
        amplitudes = np.ones(len(times))
        return V2SpikeTrain(times=times, amplitudes=amplitudes)
    
    @staticmethod
    def is_available() -> bool:
        """Check if Neo package is available."""
        try:
            import neo
            import quantities
            return True
        except ImportError:
            return False
