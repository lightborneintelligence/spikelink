"""
Nengo Adapter — SpikeLink bridge for Nengo (neuromorphic simulator).

Bridges Nengo's probe data and spike processes to SpikeLink's
spike-native transport, matching the NeoAdapter pattern:

    NengoAdapter.from_nengo(probe_data)   → SpikeLink SpikeTrain
    NengoAdapter.to_nengo(train)          → Nengo spike input process
    NengoAdapter.verify_round_trip(...)   → bool

Supports Nengo 3.x+ and NengoLoihi for hardware deployment.

Example:
    >>> import nengo
    >>> from spikelink.adapters import NengoAdapter
    >>> from spikelink import SpikeTrain, SpikelinkCodec
    >>>
    >>> with nengo.Network() as net:
    ...     ens = nengo.Ensemble(10, 1)
    ...     probe = nengo.Probe(ens.neurons)
    >>> with nengo.Simulator(net) as sim:
    ...     sim.run(1.0)
    >>>
    >>> # Extract spikes from probe data
    >>> trains = NengoAdapter.from_nengo(sim.data[probe], dt=sim.dt)
    >>>
    >>> # Transport through SpikeLink
    >>> codec = SpikelinkCodec()
    >>> for train in trains:
    ...     packets = codec.encode_train(train)

Lightborne Intelligence · Dallas TX
Truth > Consensus · Sovereignty > Control · Coherence > Speed
"""

import numpy as np
from typing import Optional, List, Dict, Union, Callable

from ..types.spiketrain import SpikeTrain


class NengoAdapter:
    """
    Adapter between Nengo probe data / processes and SpikeLink SpikeTrain.

    Follows the same static-method pattern as NeoAdapter:
        from_nengo()           → extract spikes from Nengo probe → SpikeTrain(s)
        from_nengo_spikes()    → extract from spike raster → SpikeTrain(s)
        to_nengo()             → create Nengo Node with spike input
        to_nengo_process()     → create Nengo Process for spike generation
        verify_round_trip()    → full round-trip verification
    """

    @staticmethod
    def _check_nengo():
        """Check if Nengo is available and return module."""
        try:
            import nengo
            return nengo
        except ImportError:
            raise ImportError(
                "NengoAdapter requires 'nengo' package. "
                "Install with: pip install spikelink[nengo]"
            )

    # ── Nengo → SpikeLink ────────────────────────────────────

    @staticmethod
    def from_nengo(
        probe_data: np.ndarray,
        dt: float = 0.001,
        neuron_index: Optional[int] = None,
    ) -> Union[SpikeTrain, List[SpikeTrain]]:
        """
        Extract spike times from Nengo probe data → SpikeLink SpikeTrain(s).

        Nengo spike probes return a 2D array: (n_timesteps, n_neurons)
        where values > 0 indicate spikes (typically 1/dt for rate).

        Args:
            probe_data:    Nengo probe data array (n_timesteps, n_neurons).
            dt:            Simulation timestep in seconds (default: 0.001).
            neuron_index:  If provided, extract only this neuron.
                           If None, returns list of all neurons.

        Returns:
            SpikeTrain if neuron_index specified, else List[SpikeTrain].
        """
        probe_data = np.asarray(probe_data)
        
        # Handle 1D array (single neuron)
        if probe_data.ndim == 1:
            probe_data = probe_data.reshape(-1, 1)
        
        n_timesteps, n_neurons = probe_data.shape
        
        if neuron_index is not None:
            # Extract single neuron
            return NengoAdapter._extract_neuron_spikes(
                probe_data[:, neuron_index], dt, n_timesteps
            )
        
        # Extract all neurons
        trains = []
        for i in range(n_neurons):
            train = NengoAdapter._extract_neuron_spikes(
                probe_data[:, i], dt, n_timesteps
            )
            trains.append(train)
        
        return trains

    @staticmethod
    def _extract_neuron_spikes(
        spike_trace: np.ndarray,
        dt: float,
        n_timesteps: int,
    ) -> SpikeTrain:
        """Extract spike times from a single neuron's trace."""
        # Spikes are indicated by non-zero values
        spike_indices = np.where(spike_trace > 0)[0]
        times_s = spike_indices * dt
        
        t_start = 0.0
        t_stop = n_timesteps * dt
        
        return SpikeTrain(times=times_s, t_start=t_start, t_stop=t_stop)

    @staticmethod
    def from_nengo_spikes(
        spike_times: Union[List[float], np.ndarray],
        t_start: float = 0.0,
        t_stop: Optional[float] = None,
    ) -> SpikeTrain:
        """
        Create SpikeTrain from raw spike times (already in seconds).

        Args:
            spike_times: Array of spike times in seconds.
            t_start:     Recording start time.
            t_stop:      Recording end time (inferred if None).

        Returns:
            spikelink.SpikeTrain.
        """
        times = np.asarray(spike_times, dtype=np.float64)
        times = np.sort(times)
        
        if t_stop is None:
            t_stop = float(times[-1]) + 0.1 if len(times) > 0 else 1.0
        
        return SpikeTrain(times=times, t_start=t_start, t_stop=t_stop)

    # ── SpikeLink → Nengo ────────────────────────────────────

    @staticmethod
    def to_nengo(
        train: SpikeTrain,
        amplitude: float = 1.0,
    ) -> Callable:
        """
        Create a Nengo Node function that outputs spikes at specified times.

        This returns a function suitable for use with nengo.Node:
            node = nengo.Node(NengoAdapter.to_nengo(train), size_out=1)

        The function outputs `amplitude` at spike times, 0 otherwise.
        Uses a tolerance of dt/2 for spike detection.

        Args:
            train:     SpikeLink SpikeTrain (times in seconds).
            amplitude: Output amplitude at spike times (default: 1.0).

        Returns:
            Callable for nengo.Node (t -> output).
        """
        spike_times = np.asarray(train.times, dtype=np.float64)
        
        def spike_node(t):
            # Check if current time matches any spike (within tolerance)
            if len(spike_times) == 0:
                return 0.0
            
            # Find closest spike
            diffs = np.abs(spike_times - t)
            min_diff = np.min(diffs)
            
            # Tolerance: half a typical dt (0.5 ms)
            if min_diff < 0.0005:
                return amplitude
            return 0.0
        
        return spike_node

    @staticmethod
    def to_nengo_process(
        train: SpikeTrain,
        dt: float = 0.001,
    ):
        """
        Create a Nengo Process that generates spikes at specified times.

        Returns a nengo.processes.Process subclass instance.

        Args:
            train: SpikeLink SpikeTrain (times in seconds).
            dt:    Expected simulation timestep.

        Returns:
            Nengo Process instance.
        """
        nengo = NengoAdapter._check_nengo()
        
        spike_times = np.asarray(train.times, dtype=np.float64)
        
        class SpikeProcess(nengo.processes.Process):
            def __init__(self, times, dt):
                self.spike_times = times
                self.spike_dt = dt
                super().__init__(default_size_out=1)
            
            def make_step(self, shape_in, shape_out, dt, rng, state):
                times = self.spike_times
                threshold = dt / 2
                
                def step(t):
                    if len(times) == 0:
                        return np.array([0.0])
                    diffs = np.abs(times - t)
                    if np.min(diffs) < threshold:
                        return np.array([1.0 / dt])  # Nengo spike rate convention
                    return np.array([0.0])
                
                return step
        
        return SpikeProcess(spike_times, dt)

    @staticmethod
    def to_nengo_spike_input(
        trains: List[SpikeTrain],
        dt: float = 0.001,
    ) -> np.ndarray:
        """
        Convert multiple SpikeTrains to Nengo-compatible spike raster.

        Creates a 2D array (n_timesteps, n_neurons) suitable for
        use with nengo.processes.PresentInput or similar.

        Args:
            trains: List of SpikeTrain objects.
            dt:     Timestep for discretization.

        Returns:
            np.ndarray of shape (n_timesteps, n_neurons).
        """
        if not trains:
            return np.array([]).reshape(0, 0)
        
        # Find time bounds
        t_stop = max(
            t.t_stop if hasattr(t, 't_stop') else (t.times[-1] + 0.1 if len(t.times) > 0 else 1.0)
            for t in trains
        )
        
        n_timesteps = int(np.ceil(t_stop / dt))
        n_neurons = len(trains)
        
        raster = np.zeros((n_timesteps, n_neurons), dtype=np.float32)
        
        for i, train in enumerate(trains):
            for spike_time in train.times:
                idx = int(spike_time / dt)
                if 0 <= idx < n_timesteps:
                    raster[idx, i] = 1.0 / dt  # Nengo rate convention
        
        return raster

    # ── Round-trip verification ──────────────────────────────

    @staticmethod
    def verify_round_trip(
        spike_times: Union[List[float], np.ndarray],
        dt: float = 0.001,
        tolerance_s: float = None,
    ) -> bool:
        """
        Verify round-trip: times → SpikeTrain → raster → SpikeTrain.

        Args:
            spike_times: Spike times in seconds.
            dt:          Timestep for discretization.
            tolerance_s: Timing tolerance (default: dt).

        Returns:
            True if round-trip preserves data within tolerance.
        """
        if tolerance_s is None:
            tolerance_s = dt
        
        from ..core.codec import SpikelinkCodec
        
        spike_times = np.asarray(spike_times, dtype=np.float64)
        n_spikes = len(spike_times)
        
        # Step 1: Create SpikeTrain
        original = NengoAdapter.from_nengo_spikes(spike_times)
        
        # Step 2: SpikeLink codec round-trip
        codec = SpikelinkCodec()
        packets = codec.encode_train(original)
        recovered = codec.decode_packets(packets)
        
        # Step 3: Convert to Nengo raster and back
        raster = NengoAdapter.to_nengo_spike_input([recovered], dt=dt)
        final_list = NengoAdapter.from_nengo(raster, dt=dt)
        
        if not final_list:
            return n_spikes == 0
        
        final = final_list[0]
        
        # Verify count (allow for discretization)
        if abs(len(final.times) - n_spikes) > 1:
            return False
        
        # Verify timing (within discretization tolerance)
        if n_spikes > 0 and len(final.times) > 0:
            # Match spikes greedily
            matched = 0
            final_times = list(final.times)
            for t in spike_times:
                for i, ft in enumerate(final_times):
                    if abs(t - ft) <= tolerance_s:
                        matched += 1
                        final_times.pop(i)
                        break
            
            # Allow 90% match due to discretization edge effects
            if matched < 0.9 * n_spikes:
                return False
        
        return True

    @staticmethod
    def is_available() -> bool:
        """Check if Nengo is available."""
        try:
            import nengo
            return True
        except ImportError:
            return False


# Alias for V2 compatibility
NengoAdapterV2 = NengoAdapter
