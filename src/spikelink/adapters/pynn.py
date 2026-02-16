"""
PyNN Adapter — SpikeLink bridge for PyNN (0.9+/0.10+/0.12+).

Bridges PyNN's simulator-independent neuronal network API to
SpikeLink's spike-native transport, matching the NeoAdapter pattern:

    PyNNAdapter.from_pynn(population)   → SpikeLink SpikeTrain
    PyNNAdapter.to_pynn(train, sim)     → PyNN SpikeSourceArray

Supports all PyNN backends: NEST, NEURON, Brian2, mock.

Example:
    >>> import pyNN.mock as sim
    >>> from spikelink.adapters import PyNNAdapter
    >>> from spikelink import SpikeTrain, SpikelinkCodec
    >>>
    >>> sim.setup(timestep=0.1)
    >>> pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[10.0, 20.0]))
    >>> train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    >>> # train.times is now in seconds: [0.01, 0.02]
    >>>
    >>> codec = SpikelinkCodec()
    >>> packets = codec.encode_train(train)
    >>> recovered = codec.decode_packets(packets)
    >>> sim.end()

Lightborne Intelligence · Dallas TX
Truth > Consensus · Sovereignty > Control · Coherence > Speed
"""

import numpy as np
from typing import Optional, List, Dict, Union

from ..types.spiketrain import SpikeTrain


class PyNNAdapter:
    """
    Adapter between PyNN populations/spike-trains and SpikeLink SpikeTrain.

    Follows the same static-method pattern as NeoAdapter:
        from_pynn()            → extract spikes from PyNN objects → SpikeTrain
        to_pynn()              → inject SpikeTrain → PyNN SpikeSourceArray
        from_pynn_block()      → extract all neurons → list[SpikeTrain]
        to_pynn_dict()         → convert list → {neuron_idx: times_ms}
        verify_round_trip()    → full round-trip verification
    """

    @staticmethod
    def _check_pynn():
        """Check if PyNN is available."""
        try:
            import pyNN
            return True
        except ImportError:
            raise ImportError(
                "PyNNAdapter requires 'pyNN' package. "
                "Install with: pip install spikelink[pynn]"
            )

    # ── PyNN → SpikeLink ────────────────────────────────────

    @staticmethod
    def from_pynn(
        source,
        neuron_index: int = 0,
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
    ) -> SpikeTrain:
        """
        Extract spike times from a PyNN object and return a SpikeLink SpikeTrain.

        Accepts:
            - PyNN Population (extracts from neuron at neuron_index)
            - PyNN PopulationView / Assembly
            - Raw numpy array or list of spike times (ms)

        Args:
            source:        PyNN Population or raw array of spike times (ms).
            neuron_index:  Which neuron to extract (default: 0).
            t_start:       Optional start-time override (seconds).
            t_stop:        Optional stop-time override (seconds).

        Returns:
            spikelink.SpikeTrain (times in seconds).

        Raises:
            ImportError: If spikelink is not installed.
            TypeError:   If source type is unrecognised.
            IndexError:  If neuron_index is out of range.
        """
        # Handle raw arrays directly
        if isinstance(source, (list, np.ndarray)):
            times_ms = np.asarray(source, dtype=np.float64)
            times_s = times_ms / 1000.0
            times_s = np.sort(times_s)
            
            if t_start is None:
                t_start = 0.0 if len(times_s) == 0 else float(times_s[0])
            if t_stop is None:
                t_stop = 1.0 if len(times_s) == 0 else float(times_s[-1]) + 0.1
            
            return SpikeTrain(times=times_s, t_start=t_start, t_stop=t_stop)

        # PyNN Population / PopulationView / Assembly
        spike_times_ms = PyNNAdapter._extract_spikes_pynn(source, neuron_index)
        times_s = np.asarray(spike_times_ms, dtype=np.float64) / 1000.0
        times_s = np.sort(times_s)

        if t_start is None:
            t_start = 0.0 if len(times_s) == 0 else float(times_s[0])
        if t_stop is None:
            t_stop = 1.0 if len(times_s) == 0 else float(times_s[-1]) + 0.1

        return SpikeTrain(times=times_s, t_start=t_start, t_stop=t_stop)

    @staticmethod
    def _extract_spikes_pynn(source, neuron_index: int) -> np.ndarray:
        """Extract spike times from a PyNN object (internal helper)."""
        # Try get_data() for recorded populations
        if hasattr(source, "get_data"):
            try:
                block = source.get_data()
                if hasattr(block, "segments") and len(block.segments) > 0:
                    spiketrains = block.segments[0].spiketrains
                    if neuron_index < len(spiketrains):
                        st = spiketrains[neuron_index]
                        if hasattr(st, "magnitude"):
                            return np.asarray(st.magnitude)
                        return np.asarray(st)
            except Exception:
                pass

        # Try direct cellparams access (SpikeSourceArray)
        if hasattr(source, "__getitem__"):
            try:
                cell = source[neuron_index]
                if hasattr(cell, "get_parameters"):
                    params = cell.get_parameters()
                    if "spike_times" in params:
                        return np.asarray(params["spike_times"])
                if hasattr(cell, "cellparams"):
                    if "spike_times" in cell.cellparams:
                        return np.asarray(cell.cellparams["spike_times"])
            except (IndexError, TypeError):
                pass

        # Fallback: direct spike_times attribute
        if hasattr(source, "spike_times"):
            return np.asarray(source.spike_times)

        raise TypeError(
            f"Cannot extract spike times from {type(source).__name__}. "
            "Expected PyNN Population, PopulationView, Assembly, or array."
        )

    @staticmethod
    def from_pynn_block(population) -> List[SpikeTrain]:
        """
        Extract SpikeTrains for all neurons in a PyNN Population.

        Args:
            population: PyNN Population with recorded spikes.

        Returns:
            List of SpikeTrain, one per neuron (indexed by position).
        """
        size = PyNNAdapter._get_population_size(population)
        trains = []
        for i in range(size):
            try:
                train = PyNNAdapter.from_pynn(population, neuron_index=i)
                trains.append(train)
            except (IndexError, TypeError):
                trains.append(SpikeTrain(times=[]))
        return trains

    @staticmethod
    def _get_population_size(population) -> int:
        """Get the size of a PyNN population."""
        for attr in ("size", "__len__"):
            try:
                val = getattr(population, attr)
                return val() if callable(val) else int(val)
            except (AttributeError, TypeError):
                continue
        return 0

    # ── SpikeLink → PyNN ────────────────────────────────────

    @staticmethod
    def to_pynn(
        train: SpikeTrain,
        sim_module,
        population_size: int = 1,
        label: Optional[str] = None,
    ):
        """
        Convert a SpikeLink SpikeTrain to a PyNN SpikeSourceArray Population.

        Args:
            train:           SpikeLink SpikeTrain (times in seconds).
            sim_module:      PyNN backend module (e.g. pyNN.mock, pyNN.nest).
                             Must be imported and setup() already called.
            population_size: Number of neurons in the output population
                             (all will receive the same spike times).
            label:           Optional label for the PyNN Population.

        Returns:
            PyNN Population of SpikeSourceArray neurons.

        Raises:
            ValueError: If sim_module is None.
        """
        if sim_module is None:
            raise ValueError(
                "sim_module is required (e.g. pyNN.mock, pyNN.nest). "
                "Call sim_module.setup() before calling to_pynn()."
            )

        spike_times_s = np.asarray(train.times, dtype=np.float64)
        spike_times_ms = (spike_times_s * 1000.0).tolist()

        kwargs = {
            "cellclass": sim_module.SpikeSourceArray,
            "cellparams": {"spike_times": spike_times_ms},
        }
        if label is not None:
            kwargs["label"] = label

        population = sim_module.Population(population_size, **kwargs)
        return population

    @staticmethod
    def to_pynn_dict(trains: List[SpikeTrain]) -> Dict[int, List[float]]:
        """
        Convert multiple SpikeLink SpikeTrains to a dict of PyNN-compatible
        spike time arrays (ms), keyed by neuron index.

        Args:
            trains: List of SpikeLink SpikeTrain objects.

        Returns:
            Dict[int, list] mapping neuron_index → spike_times_ms.
        """
        result = {}
        for i, train in enumerate(trains):
            times_s = np.asarray(train.times, dtype=np.float64)
            result[i] = (times_s * 1000.0).tolist()
        return result

    # ── Round-trip verification ──────────────────────────────

    @staticmethod
    def verify_round_trip(
        spike_times_ms: Union[List[float], np.ndarray],
        sim_module=None,
        tolerance_s: float = 1e-5,
    ) -> bool:
        """
        Full round-trip test: raw times → SpikeLink → PyNN → SpikeLink → verify.

        Args:
            spike_times_ms:  Spike times in milliseconds.
            sim_module:      PyNN backend module (needed for to_pynn step).
            tolerance_s:     Timing tolerance in seconds (default: 10 μs).

        Returns:
            True if round-trip preserves data within tolerance.
        """
        from ..core.codec import SpikelinkCodec

        # Step 1: Raw → SpikeTrain
        original = PyNNAdapter.from_pynn(spike_times_ms)

        # Step 2: SpikeTrain → SpikeLink packets → recovered
        codec = SpikelinkCodec()
        packets = codec.encode_train(original)
        recovered = codec.decode_packets(packets)

        # Step 3: SpikeTrain → PyNN → SpikeTrain (if sim_module provided)
        if sim_module is not None:
            pop = PyNNAdapter.to_pynn(recovered, sim_module=sim_module)
            final = PyNNAdapter.from_pynn(pop, neuron_index=0)
        else:
            final = recovered

        # Verify
        if len(original.times) != len(final.times):
            return False

        if len(original.times) == 0:
            return True

        max_diff = np.max(np.abs(original.times - final.times))
        return max_diff <= tolerance_s

    @staticmethod
    def is_available() -> bool:
        """Check if PyNN is available."""
        try:
            import pyNN
            return True
        except ImportError:
            return False


# Alias for V2 compatibility
PyNNAdapterV2 = PyNNAdapter
