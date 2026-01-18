"""
SpikeTrain — Core data type for spike sequences.

A SpikeTrain represents a sequence of spike times, optionally with
associated metadata like neuron ID, units, and time bounds.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


@dataclass
class SpikeTrain:
    """
    A sequence of spike times.

    Attributes:
        times: Spike times as a list or numpy array
        t_start: Start time of the recording window (default: 0.0)
        t_stop: End time of the recording window (default: inferred from times)
        units: Time units (default: "s" for seconds)
        neuron_id: Optional identifier for the source neuron
        metadata: Optional dictionary of additional metadata

    Example:
        >>> train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        >>> print(len(train))
        5
        >>> print(train.times)
        [0.1, 0.2, 0.3, 0.4, 0.5]
    """

    times: Union[List[float], np.ndarray]
    t_start: float = 0.0
    t_stop: Optional[float] = None
    units: str = "s"
    neuron_id: Optional[Union[int, str]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize spike times."""
        # Convert to numpy array if list
        if isinstance(self.times, list):
            self.times = np.array(self.times, dtype=np.float64)

        # Ensure times are sorted
        if len(self.times) > 1 and not np.all(np.diff(self.times) >= 0):
            self.times = np.sort(self.times)

        # Infer t_stop if not provided
        if self.t_stop is None:
            if len(self.times) > 0:
                self.t_stop = float(np.max(self.times)) + 0.1
            else:
                self.t_stop = 1.0

        # Validate bounds
        if self.t_start > self.t_stop:
            raise ValueError(
                f"t_start ({self.t_start}) must be <= t_stop ({self.t_stop})"
            )

        if len(self.times) > 0:
            if np.min(self.times) < self.t_start:
                raise ValueError(
                    f"Spike times contain values before t_start ({self.t_start})"
                )
            if np.max(self.times) > self.t_stop:
                raise ValueError(
                    f"Spike times contain values after t_stop ({self.t_stop})"
                )

    def __len__(self) -> int:
        """Return the number of spikes."""
        return len(self.times)

    def __iter__(self):
        """Iterate over spike times."""
        return iter(self.times)

    def __getitem__(self, idx):
        """Get spike time by index."""
        return self.times[idx]

    @property
    def duration(self) -> float:
        """Return the duration of the recording window."""
        return self.t_stop - self.t_start

    @property
    def firing_rate(self) -> float:
        """Return the mean firing rate in Hz (assuming units are seconds)."""
        if self.duration <= 0:
            return 0.0
        return len(self.times) / self.duration

    @property
    def isi(self) -> np.ndarray:
        """Return inter-spike intervals."""
        if len(self.times) < 2:
            return np.array([])
        return np.diff(self.times)

    def time_slice(self, t_start: float, t_stop: float) -> "SpikeTrain":
        """
        Return a new SpikeTrain containing only spikes within [t_start, t_stop].

        Args:
            t_start: Start of time slice
            t_stop: End of time slice

        Returns:
            New SpikeTrain with sliced times
        """
        mask = (self.times >= t_start) & (self.times <= t_stop)
        return SpikeTrain(
            times=self.times[mask].copy(),
            t_start=t_start,
            t_stop=t_stop,
            units=self.units,
            neuron_id=self.neuron_id,
            metadata=self.metadata.copy(),
        )

    def copy(self) -> "SpikeTrain":
        """Return a deep copy of this SpikeTrain."""
        return SpikeTrain(
            times=self.times.copy(),
            t_start=self.t_start,
            t_stop=self.t_stop,
            units=self.units,
            neuron_id=self.neuron_id,
            metadata=self.metadata.copy(),
        )

    def to_list(self) -> List[float]:
        """Return spike times as a Python list."""
        return self.times.tolist()

    def __repr__(self) -> str:
        return (
            f"SpikeTrain(n_spikes={len(self)}, "
            f"t_start={self.t_start}, t_stop={self.t_stop}, "
            f"units='{self.units}')"
        )
