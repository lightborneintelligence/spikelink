"""
NEST Adapter — SpikeLink bridge for NEST simulator (3.x+).

Bridges NEST's spike_generator / spike_recorder / NodeCollection
objects to SpikeLink's spike-native transport, matching the
NeoAdapter pattern:

    NestAdapter.from_nest(recorder)        → SpikeLink SpikeTrain
    NestAdapter.from_nest_generator(gen)   → SpikeLink SpikeTrain
    NestAdapter.to_nest(train, nest_mod)   → NEST spike_generator
    NestAdapter.verify_round_trip(...)     → bool

Supports NEST 3.x+ (NodeCollection API).

Example:
    >>> import nest
    >>> from spikelink.adapters import NestAdapter
    >>> from spikelink import SpikeTrain, SpikelinkCodec
    >>>
    >>> nest.ResetKernel()
    >>> sg = nest.Create('spike_generator', params={'spike_times': [10.0, 20.0]})
    >>> sr = nest.Create('spike_recorder')
    >>> nest.Connect(sg, sr)
    >>> nest.Simulate(100.0)
    >>>
    >>> train = NestAdapter.from_nest(sr)
    >>> codec = SpikelinkCodec()
    >>> packets = codec.encode_train(train)

Lightborne Intelligence · Dallas TX
Truth > Consensus · Sovereignty > Control · Coherence > Speed
"""

import numpy as np
from typing import Optional, List, Dict, Union

from ..types.spiketrain import SpikeTrain


class NestAdapter:
    """
    Adapter between NEST simulator objects and SpikeLink SpikeTrain.

    Follows the same static-method pattern as NeoAdapter:
        from_nest()             → extract spikes from NEST recorder → SpikeTrain
        from_nest_generator()   → extract from spike_generator → SpikeTrain
        from_nest_events()      → extract from recorder → Dict[gid, SpikeTrain]
        to_nest()               → inject SpikeTrain → NEST spike_generator
        to_nest_times()         → convert SpikeTrain → spike times in ms
        verify_round_trip()     → full round-trip verification
    """

    @staticmethod
    def _check_nest():
        """Check if NEST is available and return module."""
        try:
            import nest
            return nest
        except ImportError:
            raise ImportError(
                "NestAdapter requires 'nest' package. "
                "Install NEST simulator: https://nest-simulator.readthedocs.io/"
            )

    # ── NEST → SpikeLink ────────────────────────────────────

    @staticmethod
    def from_nest(
        recorder,
        sender_gid: Optional[int] = None,
    ) -> SpikeTrain:
        """
        Extract spike times from a NEST spike_recorder → SpikeLink SpikeTrain.

        If multiple senders recorded, use sender_gid to filter.
        If sender_gid is None and multiple senders exist, returns
        spikes from all senders merged (sorted by time).

        Args:
            recorder:    NEST NodeCollection for a spike_recorder.
            sender_gid:  Optional GID to filter a specific sender.

        Returns:
            spikelink.SpikeTrain (times in seconds).
        """
        events = NestAdapter._get_events(recorder)

        if events is None or len(events.get("times", [])) == 0:
            return SpikeTrain(times=[])

        times_ms = np.asarray(events["times"], dtype=np.float64)
        senders = np.asarray(events.get("senders", []), dtype=np.int64)

        # Filter by sender if requested
        if sender_gid is not None and len(senders) > 0:
            mask = senders == sender_gid
            times_ms = times_ms[mask]

        # Convert to seconds, sort
        times_s = times_ms / 1000.0
        times_s = np.sort(times_s)

        t_start = 0.0 if len(times_s) == 0 else float(times_s[0])
        t_stop = 1.0 if len(times_s) == 0 else float(times_s[-1]) + 0.1

        return SpikeTrain(times=times_s, t_start=t_start, t_stop=t_stop)

    @staticmethod
    def from_nest_events(recorder) -> Dict[int, SpikeTrain]:
        """
        Extract per-sender SpikeTrains from a spike_recorder.

        Returns a dict mapping sender GID → SpikeTrain.

        Args:
            recorder: NEST NodeCollection for a spike_recorder.

        Returns:
            Dict[int, SpikeTrain] for each unique sender.
        """
        events = NestAdapter._get_events(recorder)

        if events is None or len(events.get("times", [])) == 0:
            return {}

        times_ms = np.asarray(events["times"], dtype=np.float64)
        senders = np.asarray(events["senders"], dtype=np.int64)

        result = {}
        for gid in np.unique(senders):
            mask = senders == gid
            sender_times_ms = times_ms[mask]
            sender_times_s = np.sort(sender_times_ms / 1000.0)

            t_start = 0.0 if len(sender_times_s) == 0 else float(sender_times_s[0])
            t_stop = 1.0 if len(sender_times_s) == 0 else float(sender_times_s[-1]) + 0.1

            result[int(gid)] = SpikeTrain(
                times=sender_times_s, t_start=t_start, t_stop=t_stop
            )

        return result

    @staticmethod
    def from_nest_generator(generator) -> SpikeTrain:
        """
        Extract spike times directly from a NEST spike_generator.

        This extracts the configured spike_times without simulation,
        giving exact timing (no delay artifacts).

        Args:
            generator: NEST NodeCollection for a spike_generator.

        Returns:
            spikelink.SpikeTrain (times in seconds).
        """
        params = NestAdapter._get_params(generator)
        times_ms = params.get("spike_times", [])

        if isinstance(times_ms, (list, tuple)):
            times_ms = np.asarray(times_ms, dtype=np.float64)
        else:
            times_ms = np.asarray([times_ms], dtype=np.float64)

        times_s = times_ms / 1000.0
        times_s = np.sort(times_s)

        t_start = 0.0 if len(times_s) == 0 else float(times_s[0])
        t_stop = 1.0 if len(times_s) == 0 else float(times_s[-1]) + 0.1

        return SpikeTrain(times=times_s, t_start=t_start, t_stop=t_stop)

    @staticmethod
    def _get_events(recorder) -> Optional[Dict]:
        """Get events dict from a spike_recorder (internal helper)."""
        # Try .get() method (NEST 3.x)
        if hasattr(recorder, "get"):
            try:
                return recorder.get("events")
            except Exception:
                pass

        # Try GetStatus (older API)
        try:
            import nest
            status = nest.GetStatus(recorder)
            if isinstance(status, (list, tuple)) and len(status) > 0:
                return status[0].get("events", {})
            return status.get("events", {})
        except Exception:
            pass

        return None

    @staticmethod
    def _get_params(node) -> Dict:
        """Get parameters from a NEST node (internal helper)."""
        # Try .get() method (NEST 3.x)
        if hasattr(node, "get"):
            try:
                return dict(node.get())
            except Exception:
                pass

        # Try GetStatus (older API)
        try:
            import nest
            status = nest.GetStatus(node)
            if isinstance(status, (list, tuple)) and len(status) > 0:
                return dict(status[0])
            return dict(status)
        except Exception:
            pass

        return {}

    # ── SpikeLink → NEST ────────────────────────────────────

    @staticmethod
    def to_nest(
        train: SpikeTrain,
        nest_module=None,
        params: Optional[Dict] = None,
    ):
        """
        Convert a SpikeLink SpikeTrain to a NEST spike_generator.

        Args:
            train:        SpikeLink SpikeTrain (times in seconds).
            nest_module:  The 'nest' module (import nest). If None, attempts import.
            params:       Additional parameters for spike_generator.

        Returns:
            NEST NodeCollection for the created spike_generator.

        Raises:
            ImportError: If NEST is not available.
        """
        if nest_module is None:
            nest_module = NestAdapter._check_nest()

        spike_times_s = np.asarray(train.times, dtype=np.float64)
        spike_times_ms = (spike_times_s * 1000.0).tolist()

        gen_params = {"spike_times": spike_times_ms}
        if params:
            gen_params.update(params)

        return nest_module.Create("spike_generator", params=gen_params)

    @staticmethod
    def to_nest_times(train: SpikeTrain) -> List[float]:
        """
        Convert a SpikeLink SpikeTrain to NEST-compatible spike times (ms).

        Args:
            train: SpikeLink SpikeTrain (times in seconds).

        Returns:
            List of spike times in milliseconds.
        """
        times_s = np.asarray(train.times, dtype=np.float64)
        return (times_s * 1000.0).tolist()

    # ── Round-trip verification ──────────────────────────────

    @staticmethod
    def verify_round_trip(
        spike_times_ms: Union[List[float], np.ndarray],
        nest_module=None,
        tolerance_s: float = 1e-5,
    ) -> bool:
        """
        Two-phase round-trip verification:
          1. Timing fidelity via generator extraction (exact)
          2. Count preservation via recorder (simulation)

        Args:
            spike_times_ms: Spike times in milliseconds.
            nest_module:    The 'nest' module. If None, attempts import.
            tolerance_s:    Timing tolerance in seconds (default: 10 μs).

        Returns:
            True if both phases pass.
        """
        if nest_module is None:
            nest_module = NestAdapter._check_nest()

        from ..core.codec import SpikelinkCodec

        spike_times_ms = np.asarray(spike_times_ms, dtype=np.float64)
        n_spikes = len(spike_times_ms)

        # Phase 1: Generator extraction (exact timing)
        nest_module.ResetKernel()
        original_train = SpikeTrain(times=spike_times_ms / 1000.0)

        sg = NestAdapter.to_nest(original_train, nest_module=nest_module)
        extracted = NestAdapter.from_nest_generator(sg)

        if len(extracted.times) != n_spikes:
            return False

        if n_spikes > 0:
            max_diff = np.max(np.abs(original_train.times - extracted.times))
            if max_diff > tolerance_s:
                return False

        # Phase 2: SpikeLink codec round-trip
        codec = SpikelinkCodec()
        packets = codec.encode_train(original_train)
        recovered = codec.decode_packets(packets)

        if len(recovered.times) != n_spikes:
            return False

        # Phase 3: Recorder round-trip (with simulation)
        nest_module.ResetKernel()
        sg2 = NestAdapter.to_nest(recovered, nest_module=nest_module)
        sr = nest_module.Create("spike_recorder")
        nest_module.Connect(sg2, sr)

        sim_time = 100.0 if n_spikes == 0 else float(np.max(spike_times_ms)) + 10.0
        nest_module.Simulate(sim_time)

        final = NestAdapter.from_nest(sr)

        return len(final.times) == n_spikes

    @staticmethod
    def is_available() -> bool:
        """Check if NEST is available."""
        try:
            import nest
            return True
        except ImportError:
            return False


# Alias for V2 compatibility
NestAdapterV2 = NestAdapter
