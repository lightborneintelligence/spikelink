"""
Tonic (event-camera) adapter for SpikeLink.

Provides conversion between Tonic event arrays and SpikeLink formats.
Requires: pip install spikelink[tonic]
"""

import numpy as np
from ..v2.types import SpikeTrainV2


class TonicAdapter:
    """
    Bridge between Tonic event-camera data and SpikeLink v2 SpikeTrain.
    
    Tonic events: structured array with (x, y, t, p) fields
      - x, y: pixel coordinates
      - t: timestamp (microseconds typically)
      - p: polarity (0 or 1)
    
    v2 SpikeTrain: times + amplitudes
    
    Strategy:
      Tonic → v2: Multiple modes depending on use case
        - Mode A (timing-exact): spikes_per_packet=1, preserves 10μs timing
        - Mode B (throughput): larger chunks, bounded distortion
      
      Each (x, y, p) combination can be treated as a separate "channel"
      or all events can be flattened into a single stream.
    """
    
    @staticmethod
    def _check_tonic():
        try:
            import tonic
            return tonic
        except ImportError:
            raise ImportError(
                "TonicAdapter requires 'tonic' package. "
                "Install with: pip install spikelink[tonic]"
            )
    
    @staticmethod
    def from_events(events, time_unit: str = 'us',
                    default_amplitude: float = 1.0) -> SpikeTrainV2:
        """
        Convert Tonic event array → v2 SpikeTrain (flattened).
        
        Args:
            events: Structured numpy array with 't' field (timestamps)
            time_unit: 'us' (microseconds), 'ms', or 's'
            default_amplitude: Amplitude to assign to all events
        
        Returns:
            SpikeTrainV2: All events as a single spike train
        """
        times = np.array(events['t'], dtype=np.float64)
        
        # Convert to seconds
        if time_unit == 'us':
            times = times * 1e-6
        elif time_unit == 'ms':
            times = times * 1e-3
        # else assume seconds
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        
        amplitudes = np.full(len(times), default_amplitude)
        return SpikeTrainV2(times=times, amplitudes=amplitudes)
    
    @staticmethod
    def from_events_by_channel(events, time_unit: str = 'us',
                               default_amplitude: float = 1.0) -> dict:
        """
        Convert Tonic events → dict of v2 SpikeTrains keyed by (x, y, p).
        
        Args:
            events: Structured numpy array with x, y, t, p fields
            time_unit: 'us', 'ms', or 's'
            default_amplitude: Amplitude to assign
        
        Returns:
            dict: {(x, y, p): SpikeTrainV2} for each unique channel
        """
        times = np.array(events['t'], dtype=np.float64)
        x = np.array(events['x'])
        y = np.array(events['y'])
        p = np.array(events['p'])
        
        # Convert times to seconds
        if time_unit == 'us':
            times = times * 1e-6
        elif time_unit == 'ms':
            times = times * 1e-3
        
        # Group by channel
        channels = {}
        unique_channels = set(zip(x, y, p))
        
        for (cx, cy, cp) in unique_channels:
            mask = (x == cx) & (y == cy) & (p == cp)
            ch_times = np.sort(times[mask])
            ch_amps = np.full(len(ch_times), default_amplitude)
            channels[(cx, cy, cp)] = SpikeTrainV2(times=ch_times, amplitudes=ch_amps)
        
        return channels
    
    @staticmethod
    def to_events(v2_train: SpikeTrainV2, 
                  x: int = 0, y: int = 0, p: int = 1,
                  time_unit: str = 'us') -> np.ndarray:
        """
        Convert v2 SpikeTrain → Tonic-compatible event array.
        
        Args:
            v2_train: SpikeLink v2 spike train
            x, y: Pixel coordinates to assign
            p: Polarity to assign (0 or 1)
            time_unit: Output time unit ('us', 'ms', 's')
        
        Returns:
            np.ndarray: Structured array with x, y, t, p fields
        """
        times = v2_train.times.copy()
        
        # Convert from seconds to target unit
        if time_unit == 'us':
            times = times * 1e6
        elif time_unit == 'ms':
            times = times * 1e3
        
        n = len(times)
        dtype = [('x', '<i4'), ('y', '<i4'), ('t', '<i8'), ('p', '<i4')]
        events = np.zeros(n, dtype=dtype)
        events['x'] = x
        events['y'] = y
        events['t'] = times.astype(np.int64)
        events['p'] = p
        
        return events
    
    @staticmethod
    def is_available() -> bool:
        """Check if Tonic package is available."""
        try:
            import tonic
            return True
        except ImportError:
            return False


class TonicAdapterV2(TonicAdapter):
    """Alias for TonicAdapter (v2 is the default)."""
    pass
