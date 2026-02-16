"""
Lava Adapter — SpikeLink bridge for Intel's Lava framework (Loihi).

Bridges Lava's spike tensors and processes to SpikeLink's
spike-native transport:

    LavaAdapter.from_lava(spike_tensor)   → SpikeLink SpikeTrain(s)
    LavaAdapter.to_lava(train)            → Lava-compatible spike tensor
    LavaAdapter.to_lava_input()           → Lava SpikeIn process data

Supports Lava 0.5+ for both simulation and Loihi hardware deployment.

Example:
    >>> import numpy as np
    >>> from spikelink.adapters import LavaAdapter
    >>> from spikelink import SpikeTrain, SpikelinkCodec
    >>>
    >>> # Lava spike tensor: (timesteps, neurons), 1 = spike
    >>> spike_tensor = np.zeros((100, 10))
    >>> spike_tensor[10, 0] = 1  # Neuron 0 spikes at t=10
    >>> spike_tensor[25, 0] = 1  # Neuron 0 spikes at t=25
    >>>
    >>> # Extract to SpikeLink
    >>> trains = LavaAdapter.from_lava(spike_tensor, dt=0.001)
    >>>
    >>> # Transport through SpikeLink
    >>> codec = SpikelinkCodec()
    >>> packets = codec.encode_train(trains[0])

Lightborne Intelligence · Dallas TX
Truth > Consensus · Sovereignty > Control · Coherence > Speed
"""

import numpy as np
from typing import Optional, List, Dict, Union, Tuple

from ..types.spiketrain import SpikeTrain


class LavaAdapter:
    """
    Adapter between Lava spike tensors and SpikeLink SpikeTrain.

    Lava uses binary spike tensors with shape:
        (timesteps, neurons)           - 2D standard
        (batch, timesteps, neurons)    - 3D batched

    Follows the same static-method pattern as NeoAdapter:
        from_lava()            → extract spikes from tensor → SpikeTrain(s)
        from_lava_events()     → extract from sparse events → SpikeTrain(s)
        to_lava()              → convert SpikeTrain → spike tensor
        to_lava_sparse()       → convert SpikeTrain → sparse event format
        verify_round_trip()    → full round-trip verification
    """

    @staticmethod
    def _check_lava():
        """Check if Lava is available and return module."""
        try:
            import lava
            return lava
        except ImportError:
            raise ImportError(
                "LavaAdapter requires 'lava-nc' package. "
                "Install with: pip install lava-nc"
            )

    # ── Lava → SpikeLink ────────────────────────────────────

    @staticmethod
    def from_lava(
        spike_tensor: np.ndarray,
        dt: float = 0.001,
        neuron_index: Optional[int] = None,
        batch_index: int = 0,
    ) -> Union[SpikeTrain, List[SpikeTrain]]:
        """
        Extract spike times from Lava spike tensor → SpikeLink SpikeTrain(s).

        Args:
            spike_tensor:  Binary spike tensor.
                           Shape: (timesteps, neurons) or (batch, timesteps, neurons)
            dt:            Timestep in seconds (default: 0.001 = 1ms).
            neuron_index:  If provided, extract only this neuron.
                           If None, returns list of all neurons.
            batch_index:   For 3D tensors, which batch to extract (default: 0).

        Returns:
            SpikeTrain if neuron_index specified, else List[SpikeTrain].
        """
        spike_tensor = np.asarray(spike_tensor)
        
        # Handle 3D batched tensor
        if spike_tensor.ndim == 3:
            spike_tensor = spike_tensor[batch_index]
        
        # Handle 1D single neuron
        if spike_tensor.ndim == 1:
            spike_tensor = spike_tensor.reshape(-1, 1)
        
        n_timesteps, n_neurons = spike_tensor.shape
        t_stop = n_timesteps * dt
        
        if neuron_index is not None:
            # Extract single neuron
            spike_indices = np.where(spike_tensor[:, neuron_index] > 0)[0]
            times_s = spike_indices.astype(np.float64) * dt
            return SpikeTrain(times=times_s, t_start=0.0, t_stop=t_stop)
        
        # Extract all neurons
        trains = []
        for i in range(n_neurons):
            spike_indices = np.where(spike_tensor[:, i] > 0)[0]
            times_s = spike_indices.astype(np.float64) * dt
            trains.append(SpikeTrain(times=times_s, t_start=0.0, t_stop=t_stop))
        
        return trains

    @staticmethod
    def from_lava_events(
        times: np.ndarray,
        neurons: np.ndarray,
        dt: float = 0.001,
        n_neurons: Optional[int] = None,
    ) -> Dict[int, SpikeTrain]:
        """
        Extract spike times from Lava sparse event format.

        Lava can represent spikes as parallel arrays of (time_idx, neuron_idx).

        Args:
            times:     Array of timestep indices where spikes occurred.
            neurons:   Array of neuron indices that spiked (same length as times).
            dt:        Timestep in seconds.
            n_neurons: Total number of neurons (for t_stop calculation).

        Returns:
            Dict[int, SpikeTrain] mapping neuron_id → SpikeTrain.
        """
        times = np.asarray(times, dtype=np.int64)
        neurons = np.asarray(neurons, dtype=np.int64)
        
        if len(times) != len(neurons):
            raise ValueError("times and neurons arrays must have same length")
        
        if n_neurons is None:
            n_neurons = int(np.max(neurons)) + 1 if len(neurons) > 0 else 0
        
        t_stop = float(np.max(times) + 1) * dt if len(times) > 0 else 1.0
        
        result = {}
        for neuron_id in np.unique(neurons):
            mask = neurons == neuron_id
            spike_times_s = times[mask].astype(np.float64) * dt
            spike_times_s = np.sort(spike_times_s)
            result[int(neuron_id)] = SpikeTrain(
                times=spike_times_s, t_start=0.0, t_stop=t_stop
            )
        
        return result

    # ── SpikeLink → Lava ────────────────────────────────────

    @staticmethod
    def to_lava(
        train: SpikeTrain,
        dt: float = 0.001,
        n_timesteps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert SpikeLink SpikeTrain to Lava spike tensor.

        Args:
            train:        SpikeLink SpikeTrain.
            dt:           Timestep in seconds.
            n_timesteps:  Number of timesteps (inferred from t_stop if None).

        Returns:
            Binary spike tensor of shape (n_timesteps,).
        """
        times_s = np.asarray(train.times, dtype=np.float64)
        
        if n_timesteps is None:
            t_stop = train.t_stop if hasattr(train, 't_stop') else (
                float(times_s[-1]) + dt if len(times_s) > 0 else 1.0
            )
            n_timesteps = int(np.ceil(t_stop / dt))
        
        tensor = np.zeros(n_timesteps, dtype=np.int8)
        
        for t in times_s:
            idx = int(t / dt)
            if 0 <= idx < n_timesteps:
                tensor[idx] = 1
        
        return tensor

    @staticmethod
    def to_lava_batch(
        trains: List[SpikeTrain],
        dt: float = 0.001,
        n_timesteps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert multiple SpikeTrains to Lava spike tensor.

        Args:
            trains:       List of SpikeTrain objects.
            dt:           Timestep in seconds.
            n_timesteps:  Number of timesteps (inferred if None).

        Returns:
            Binary spike tensor of shape (n_timesteps, n_neurons).
        """
        if not trains:
            return np.array([]).reshape(0, 0)
        
        if n_timesteps is None:
            t_stop = max(
                t.t_stop if hasattr(t, 't_stop') else (
                    float(t.times[-1]) + dt if len(t.times) > 0 else 1.0
                )
                for t in trains
            )
            n_timesteps = int(np.ceil(t_stop / dt))
        
        n_neurons = len(trains)
        tensor = np.zeros((n_timesteps, n_neurons), dtype=np.int8)
        
        for i, train in enumerate(trains):
            for t in train.times:
                idx = int(t / dt)
                if 0 <= idx < n_timesteps:
                    tensor[idx, i] = 1
        
        return tensor

    @staticmethod
    def to_lava_sparse(
        train: SpikeTrain,
        dt: float = 0.001,
        neuron_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert SpikeTrain to Lava sparse event format.

        Args:
            train:      SpikeLink SpikeTrain.
            dt:         Timestep in seconds.
            neuron_id:  Neuron ID to assign to all spikes.

        Returns:
            Tuple of (time_indices, neuron_indices).
        """
        times_s = np.asarray(train.times, dtype=np.float64)
        time_indices = (times_s / dt).astype(np.int64)
        neuron_indices = np.full(len(time_indices), neuron_id, dtype=np.int64)
        
        return time_indices, neuron_indices

    @staticmethod
    def to_lava_sparse_batch(
        trains: List[SpikeTrain],
        dt: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert multiple SpikeTrains to Lava sparse event format.

        Args:
            trains: List of SpikeTrain objects.
            dt:     Timestep in seconds.

        Returns:
            Tuple of (time_indices, neuron_indices) for all spikes.
        """
        all_times = []
        all_neurons = []
        
        for i, train in enumerate(trains):
            times_s = np.asarray(train.times, dtype=np.float64)
            time_indices = (times_s / dt).astype(np.int64)
            neuron_indices = np.full(len(time_indices), i, dtype=np.int64)
            
            all_times.append(time_indices)
            all_neurons.append(neuron_indices)
        
        if not all_times:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        return np.concatenate(all_times), np.concatenate(all_neurons)

    # ── Round-trip verification ──────────────────────────────

    @staticmethod
    def verify_round_trip(
        spike_times: Union[List[float], np.ndarray],
        dt: float = 0.001,
        tolerance_steps: int = 1,
    ) -> bool:
        """
        Verify round-trip: times → SpikeTrain → tensor → SpikeTrain.

        Args:
            spike_times:     Spike times in seconds.
            dt:              Timestep for discretization.
            tolerance_steps: Allowed timing error in timesteps.

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
        
        # Step 3: Convert to Lava tensor and back
        tensor = LavaAdapter.to_lava(recovered, dt=dt)
        final_list = LavaAdapter.from_lava(tensor.reshape(-1, 1), dt=dt)
        
        if not final_list:
            return n_spikes == 0
        
        final = final_list[0]
        
        # Verify count
        if len(final.times) != n_spikes:
            return False
        
        # Verify timing (within discretization tolerance)
        if n_spikes > 0:
            tolerance_s = tolerance_steps * dt
            for orig_t, final_t in zip(np.sort(spike_times), np.sort(final.times)):
                if abs(orig_t - final_t) > tolerance_s:
                    return False
        
        return True

    @staticmethod
    def is_available() -> bool:
        """Check if Lava is available."""
        try:
            import lava
            return True
        except ImportError:
            return False


# Alias for V2 compatibility
LavaAdapterV2 = LavaAdapter
