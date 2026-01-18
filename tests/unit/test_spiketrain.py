"""Unit tests for SpikeTrain."""

import numpy as np
import pytest

from spikelink import SpikeTrain


class TestSpikeTrain:
    """Tests for the SpikeTrain class."""

    def test_create_from_list(self):
        """Test creation from Python list."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])

        assert len(train) == 3
        assert isinstance(train.times, np.ndarray)

    def test_create_from_array(self):
        """Test creation from numpy array."""
        times = np.array([0.1, 0.2, 0.3])
        train = SpikeTrain(times=times)

        assert len(train) == 3

    def test_create_empty(self):
        """Test creation of empty spike train."""
        train = SpikeTrain(times=[])

        assert len(train) == 0

    def test_auto_sort(self):
        """Test that unsorted times are automatically sorted."""
        train = SpikeTrain(times=[0.3, 0.1, 0.2])

        np.testing.assert_array_equal(train.times, [0.1, 0.2, 0.3])

    def test_auto_t_stop(self):
        """Test that t_stop is auto-inferred."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])

        assert train.t_stop >= 0.3

    def test_custom_bounds(self):
        """Test custom time bounds."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3], t_start=0.0, t_stop=1.0)

        assert train.t_start == 0.0
        assert train.t_stop == 1.0

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="t_start.*t_stop"):
            SpikeTrain(times=[0.5], t_start=1.0, t_stop=0.5)

    def test_times_before_t_start(self):
        """Test that times before t_start raise error."""
        with pytest.raises(ValueError, match="before t_start"):
            SpikeTrain(times=[0.1, 0.2], t_start=0.5, t_stop=1.0)

    def test_times_after_t_stop(self):
        """Test that times after t_stop raise error."""
        with pytest.raises(ValueError, match="after t_stop"):
            SpikeTrain(times=[0.1, 0.9], t_start=0.0, t_stop=0.5)


class TestSpikeTrainProperties:
    """Tests for SpikeTrain computed properties."""

    def test_duration(self):
        """Test duration property."""
        train = SpikeTrain(times=[0.1, 0.2], t_start=0.0, t_stop=1.0)

        assert train.duration == 1.0

    def test_firing_rate(self):
        """Test firing rate calculation."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5], t_start=0.0, t_stop=1.0)

        assert train.firing_rate == 5.0  # 5 spikes in 1 second

    def test_isi(self):
        """Test inter-spike interval calculation."""
        train = SpikeTrain(times=[0.1, 0.2, 0.4, 0.5])

        expected_isi = np.array([0.1, 0.2, 0.1])
        np.testing.assert_array_almost_equal(train.isi, expected_isi)

    def test_isi_empty(self):
        """Test ISI for empty/single spike train."""
        empty = SpikeTrain(times=[])
        single = SpikeTrain(times=[0.5])

        assert len(empty.isi) == 0
        assert len(single.isi) == 0


class TestSpikeTrainMethods:
    """Tests for SpikeTrain methods."""

    def test_time_slice(self):
        """Test time slicing."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5], t_start=0.0, t_stop=1.0)

        sliced = train.time_slice(0.15, 0.35)

        assert len(sliced) == 2  # 0.2 and 0.3
        assert sliced.t_start == 0.15
        assert sliced.t_stop == 0.35

    def test_copy(self):
        """Test deep copy."""
        original = SpikeTrain(times=[0.1, 0.2, 0.3], metadata={"key": "value"})
        copied = original.copy()

        # Modify original
        original.times[0] = 0.5
        original.metadata["key"] = "modified"

        # Copy should be unchanged
        assert copied.times[0] == 0.1
        assert copied.metadata["key"] == "value"

    def test_to_list(self):
        """Test conversion to Python list."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])

        times_list = train.to_list()

        assert isinstance(times_list, list)
        assert times_list == [0.1, 0.2, 0.3]

    def test_iteration(self):
        """Test iteration over spike times."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])

        times = [t for t in train]

        assert times == [0.1, 0.2, 0.3]

    def test_indexing(self):
        """Test indexing spike times."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])

        assert train[0] == 0.1
        assert train[1] == 0.2
        assert train[-1] == 0.3
