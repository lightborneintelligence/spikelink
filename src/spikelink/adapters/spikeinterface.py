"""
SpikeInterface Adapter — SpikeLink bridge for electrophysiology data.

Bridges SpikeInterface's sorting extractors and spike sorting results
to SpikeLink's spike-native transport:

    SpikeInterfaceAdapter.from_sorting(sorting)  → Dict[unit_id, SpikeTrain]
    SpikeInterfaceAdapter.to_sorting(trains)     → NumpySorting object
    SpikeInterfaceAdapter.verify_round_trip()    → bool

Compatible with all SpikeInterface spike sorters:
    KiloSort, SpyKING CIRCUS, MountainSort, Tridesclous, etc.

Example:
    >>> import spikeinterface.extractors as se
    >>> from spikelink.adapters import SpikeInterfaceAdapter
    >>> from spikelink import SpikelinkCodec
    >>>
    >>> # Load spike sorting results
    >>> sorting = se.read_kilosort('/path/to/kilosort/output')
    >>>
    >>> # Extract all units to SpikeLink
    >>> trains = SpikeInterfaceAdapter.from_sorting(sorting)
    >>>
    >>> # Transport through SpikeLink
    >>> codec = SpikelinkCodec()
    >>> for unit_id, train in trains.items():
    ...     packets = codec.encode_train(train)
    ...     print(f"Unit {unit_id}: {len(train.times)} spikes → {len(packets)} packets")

Lightborne Intelligence · Dallas TX
Truth > Consensus · Sovereignty > Control · Coherence > Speed
"""

import numpy as np
from typing import Optional, List, Dict, Union

from ..types.spiketrain import SpikeTrain


class SpikeInterfaceAdapter:
    """
    Adapter between SpikeInterface sorting objects and SpikeLink SpikeTrain.

    SpikeInterface is the standard library for electrophysiology data,
    providing unified access to spike sorting results from all major sorters.

    Follows the same static-method pattern as NeoAdapter:
        from_sorting()        → extract all units → Dict[unit_id, SpikeTrain]
        from_unit()           → extract single unit → SpikeTrain
        to_sorting()          → create NumpySorting from SpikeTrains
        to_spike_vector()     → create spike_vector format
        verify_round_trip()   → full round-trip verification
    """

    @staticmethod
    def _check_spikeinterface():
        """Check if SpikeInterface is available and return module."""
        try:
            import spikeinterface
            return spikeinterface
        except ImportError:
            raise ImportError(
                "SpikeInterfaceAdapter requires 'spikeinterface' package. "
                "Install with: pip install spikelink[spikeinterface]"
            )

    # ── SpikeInterface → SpikeLink ────────────────────────────

    @staticmethod
    def from_sorting(
        sorting,
        segment_index: int = 0,
    ) -> Dict[Union[int, str], SpikeTrain]:
        """
        Extract all units from a SpikeInterface sorting → Dict of SpikeTrains.

        Args:
            sorting:       SpikeInterface BaseSorting or SortingExtractor.
            segment_index: For multi-segment recordings (default: 0).

        Returns:
            Dict mapping unit_id → SpikeTrain for each sorted unit.
        """
        result = {}
        
        # Get sampling frequency
        fs = SpikeInterfaceAdapter._get_sampling_frequency(sorting)
        
        # Get unit IDs
        unit_ids = SpikeInterfaceAdapter._get_unit_ids(sorting)
        
        for unit_id in unit_ids:
            train = SpikeInterfaceAdapter.from_unit(
                sorting, unit_id, segment_index=segment_index
            )
            result[unit_id] = train
        
        return result

    @staticmethod
    def from_unit(
        sorting,
        unit_id: Union[int, str],
        segment_index: int = 0,
    ) -> SpikeTrain:
        """
        Extract a single unit from a SpikeInterface sorting → SpikeTrain.

        Args:
            sorting:       SpikeInterface BaseSorting or SortingExtractor.
            unit_id:       ID of the unit to extract.
            segment_index: For multi-segment recordings (default: 0).

        Returns:
            spikelink.SpikeTrain (times in seconds).
        """
        # Get sampling frequency
        fs = SpikeInterfaceAdapter._get_sampling_frequency(sorting)
        
        # Get spike train (in samples)
        spike_samples = SpikeInterfaceAdapter._get_unit_spike_train(
            sorting, unit_id, segment_index
        )
        
        # Convert samples to seconds
        times_s = np.asarray(spike_samples, dtype=np.float64) / fs
        times_s = np.sort(times_s)
        
        # Get recording duration if available
        t_stop = SpikeInterfaceAdapter._get_total_duration(sorting, segment_index, fs)
        if t_stop is None:
            t_stop = float(times_s[-1]) + 0.1 if len(times_s) > 0 else 1.0
        
        return SpikeTrain(times=times_s, t_start=0.0, t_stop=t_stop)

    @staticmethod
    def from_spike_vector(
        spike_vector: np.ndarray,
        sampling_frequency: float,
    ) -> Dict[int, SpikeTrain]:
        """
        Extract SpikeTrains from SpikeInterface spike_vector format.

        spike_vector is a structured array with fields:
            'sample_index': spike times in samples
            'unit_index': unit/neuron index

        Args:
            spike_vector:        Structured numpy array with spike data.
            sampling_frequency:  Samples per second.

        Returns:
            Dict mapping unit_index → SpikeTrain.
        """
        if len(spike_vector) == 0:
            return {}
        
        samples = spike_vector['sample_index']
        units = spike_vector['unit_index']
        
        result = {}
        for unit_id in np.unique(units):
            mask = units == unit_id
            unit_samples = samples[mask]
            times_s = np.sort(unit_samples.astype(np.float64) / sampling_frequency)
            
            t_stop = float(times_s[-1]) + 0.1 if len(times_s) > 0 else 1.0
            result[int(unit_id)] = SpikeTrain(times=times_s, t_start=0.0, t_stop=t_stop)
        
        return result

    @staticmethod
    def _get_sampling_frequency(sorting) -> float:
        """Get sampling frequency from sorting object."""
        if hasattr(sorting, 'sampling_frequency'):
            return float(sorting.sampling_frequency)
        if hasattr(sorting, 'get_sampling_frequency'):
            return float(sorting.get_sampling_frequency())
        # Default to 30 kHz (common for extracellular recordings)
        return 30000.0

    @staticmethod
    def _get_unit_ids(sorting) -> List:
        """Get unit IDs from sorting object."""
        if hasattr(sorting, 'unit_ids'):
            return list(sorting.unit_ids)
        if hasattr(sorting, 'get_unit_ids'):
            return list(sorting.get_unit_ids())
        return []

    @staticmethod
    def _get_unit_spike_train(sorting, unit_id, segment_index: int = 0) -> np.ndarray:
        """Get spike train for a unit (in samples)."""
        if hasattr(sorting, 'get_unit_spike_train'):
            return np.asarray(sorting.get_unit_spike_train(unit_id, segment_index=segment_index))
        # Legacy API
        if hasattr(sorting, 'get_unit_spike_trains'):
            trains = sorting.get_unit_spike_trains()
            return np.asarray(trains.get(unit_id, []))
        return np.array([])

    @staticmethod
    def _get_total_duration(sorting, segment_index: int, fs: float) -> Optional[float]:
        """Get total recording duration in seconds."""
        if hasattr(sorting, 'get_total_duration'):
            return float(sorting.get_total_duration())
        if hasattr(sorting, 'get_num_samples'):
            try:
                n_samples = sorting.get_num_samples(segment_index)
                return float(n_samples) / fs
            except Exception:
                pass
        return None

    # ── SpikeLink → SpikeInterface ────────────────────────────

    @staticmethod
    def to_sorting(
        trains: Dict[Union[int, str], SpikeTrain],
        sampling_frequency: float = 30000.0,
    ):
        """
        Create a SpikeInterface NumpySorting from SpikeTrains.

        Args:
            trains:              Dict mapping unit_id → SpikeTrain.
            sampling_frequency:  Samples per second (default: 30 kHz).

        Returns:
            spikeinterface.core.NumpySorting object.
        """
        si = SpikeInterfaceAdapter._check_spikeinterface()
        from spikeinterface.core import NumpySorting
        
        # Convert to samples
        spike_trains_samples = {}
        for unit_id, train in trains.items():
            times_s = np.asarray(train.times, dtype=np.float64)
            samples = (times_s * sampling_frequency).astype(np.int64)
            spike_trains_samples[unit_id] = samples
        
        return NumpySorting.from_dict(
            [spike_trains_samples],
            sampling_frequency=sampling_frequency,
        )

    @staticmethod
    def to_spike_vector(
        trains: Dict[Union[int, str], SpikeTrain],
        sampling_frequency: float = 30000.0,
    ) -> np.ndarray:
        """
        Convert SpikeTrains to SpikeInterface spike_vector format.

        Args:
            trains:              Dict mapping unit_id → SpikeTrain.
            sampling_frequency:  Samples per second.

        Returns:
            Structured numpy array with 'sample_index' and 'unit_index' fields.
        """
        all_samples = []
        all_units = []
        
        for unit_id, train in trains.items():
            times_s = np.asarray(train.times, dtype=np.float64)
            samples = (times_s * sampling_frequency).astype(np.int64)
            
            all_samples.append(samples)
            all_units.append(np.full(len(samples), unit_id, dtype=np.int64))
        
        if not all_samples:
            dtype = [('sample_index', np.int64), ('unit_index', np.int64)]
            return np.array([], dtype=dtype)
        
        samples = np.concatenate(all_samples)
        units = np.concatenate(all_units)
        
        # Sort by sample index
        sort_idx = np.argsort(samples)
        samples = samples[sort_idx]
        units = units[sort_idx]
        
        dtype = [('sample_index', np.int64), ('unit_index', np.int64)]
        result = np.zeros(len(samples), dtype=dtype)
        result['sample_index'] = samples
        result['unit_index'] = units
        
        return result

    @staticmethod
    def to_times_dict(
        trains: Dict[Union[int, str], SpikeTrain],
    ) -> Dict[Union[int, str], np.ndarray]:
        """
        Convert SpikeTrains to simple dict of spike times (in seconds).

        Useful for plotting and analysis pipelines that expect raw times.

        Args:
            trains: Dict mapping unit_id → SpikeTrain.

        Returns:
            Dict mapping unit_id → numpy array of times in seconds.
        """
        return {
            unit_id: np.asarray(train.times, dtype=np.float64)
            for unit_id, train in trains.items()
        }

    # ── Round-trip verification ──────────────────────────────

    @staticmethod
    def verify_round_trip(
        spike_times: Union[List[float], np.ndarray],
        sampling_frequency: float = 30000.0,
        tolerance_samples: int = 1,
    ) -> bool:
        """
        Verify round-trip: times → SpikeTrain → sorting → SpikeTrain.

        Args:
            spike_times:         Spike times in seconds.
            sampling_frequency:  Samples per second.
            tolerance_samples:   Allowed timing error in samples.

        Returns:
            True if round-trip preserves data within tolerance.
        """
        from ..core.codec import SpikelinkCodec
        
        spike_times = np.asarray(spike_times, dtype=np.float64)
        n_spikes = len(spike_times)
        
        # Step 1: Create SpikeTrain
        original = SpikeTrain(times=spike_times)
        
        # Step 2: SpikeLink codec round-trip
        codec = SpikelinkCodec()
        packets = codec.encode_train(original)
        recovered = codec.decode_packets(packets)
        
        # Step 3: Convert to spike_vector and back
        trains_dict = {0: recovered}
        spike_vector = SpikeInterfaceAdapter.to_spike_vector(
            trains_dict, sampling_frequency=sampling_frequency
        )
        final_dict = SpikeInterfaceAdapter.from_spike_vector(
            spike_vector, sampling_frequency=sampling_frequency
        )
        
        if 0 not in final_dict:
            return n_spikes == 0
        
        final = final_dict[0]
        
        # Verify count
        if len(final.times) != n_spikes:
            return False
        
        # Verify timing (within sample tolerance)
        if n_spikes > 0:
            tolerance_s = tolerance_samples / sampling_frequency
            for orig_t, final_t in zip(np.sort(spike_times), np.sort(final.times)):
                if abs(orig_t - final_t) > tolerance_s:
                    return False
        
        return True

    @staticmethod
    def is_available() -> bool:
        """Check if SpikeInterface is available."""
        try:
            import spikeinterface
            return True
        except ImportError:
            return False


# Alias for V2 compatibility
SpikeInterfaceAdapterV2 = SpikeInterfaceAdapter
