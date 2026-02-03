"""
Brian2 adapter for SpikeLink.

Provides bidirectional conversion between Brian2 SpikeMonitor and SpikeLink formats.
Requires: pip install spikelink[brian2]
"""

import numpy as np
from ..v2.types import V2SpikeTrain


class Brian2Adapter:
    """
    Bridge between Brian2 SpikeMonitor and SpikeLink v2 SpikeTrain.
    
    Brian2 SpikeMonitor: indices (neuron IDs) + times
    v2 SpikeTrain: times + amplitudes per neuron
    
    Strategy:
      SpikeMonitor → v2: extract per-neuron spike trains, assign unit amplitudes
      v2 → Brian2: not directly supported (Brian2 generates, doesn't consume)
    """
    
    @staticmethod
    def _check_brian2():
        try:
            import brian2
            return brian2
        except ImportError:
            raise ImportError(
                "Brian2Adapter requires 'brian2' package. "
                "Install with: pip install spikelink[brian2]"
            )
    
    @staticmethod
    def from_spike_monitor(monitor, neuron_index: int = None, 
                           default_amplitude: float = 1.0) -> V2SpikeTrain:
        """
        Convert Brian2 SpikeMonitor → v2 SpikeTrain.
        
        Args:
            monitor: Brian2 SpikeMonitor object
            neuron_index: If provided, extract only this neuron's spikes.
                          If None, extracts all spikes (flattened).
            default_amplitude: Amplitude to assign to all spikes.
        
        Returns:
            V2SpikeTrain: Wave-enhanced spike train
        """
        brian2 = Brian2Adapter._check_brian2()
        
        indices = np.array(monitor.i)
        times = np.array(monitor.t)  # Already in seconds (Brian2 default)
        
        if neuron_index is not None:
            mask = indices == neuron_index
            times = times[mask]
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        
        amplitudes = np.full(len(times), default_amplitude)
        return V2SpikeTrain(times=times, amplitudes=amplitudes)
    
    @staticmethod
    def from_spike_monitor_population(monitor, 
                                      default_amplitude: float = 1.0) -> list:
        """
        Convert Brian2 SpikeMonitor → list of v2 SpikeTrains (one per neuron).
        
        Args:
            monitor: Brian2 SpikeMonitor object
            default_amplitude: Amplitude to assign to all spikes.
        
        Returns:
            list[V2SpikeTrain]: List of spike trains, indexed by neuron ID
        """
        brian2 = Brian2Adapter._check_brian2()
        
        indices = np.array(monitor.i)
        times = np.array(monitor.t)
        
        if len(indices) == 0:
            return []
        
        n_neurons = int(np.max(indices)) + 1
        trains = []
        
        for neuron_idx in range(n_neurons):
            mask = indices == neuron_idx
            neuron_times = np.sort(times[mask])
            amplitudes = np.full(len(neuron_times), default_amplitude)
            trains.append(V2SpikeTrain(times=neuron_times, amplitudes=amplitudes))
        
        return trains
    
    @staticmethod
    def is_available() -> bool:
        """Check if Brian2 package is available."""
        try:
            import brian2
            return True
        except ImportError:
            return False


class Brian2AdapterV2(Brian2Adapter):
    """Alias for Brian2Adapter (v2 is the default)."""
    pass
