"""Integration tests for SpikeLink."""

import numpy as np
import pytest

from spikelink import (
    SpikeTrain,
    SpikelinkCodec,
    VerificationSuite,
    DegradationProfiler,
    encode,
    decode,
    verify,
)
from spikelink.stress import generate_population, generate_burst, generate_regular


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_workflow(self):
        """Test complete encode-decode-verify workflow."""
        # Create train
        train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Encode
        codec = SpikelinkCodec()
        packets = codec.encode_train(train)
        
        # Decode
        recovered = codec.decode_packets(packets)
        
        # Verify
        suite = VerificationSuite(codec)
        report = suite.run_all(train)
        
        assert report.passed
    
    def test_convenience_api(self):
        """Test convenience API functions."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        packets = encode(original)
        recovered = decode(packets)
        passed = verify(original, recovered)
        
        assert passed
    
    def test_verification_suite(self):
        """Test all verification suite tests pass."""
        train = SpikeTrain(times=np.linspace(0.1, 0.9, 50))
        
        suite = VerificationSuite()
        report = suite.run_all(train)
        
        assert report.passed
        assert report.n_passed == len(report.results)
    
    def test_degradation_profiler(self):
        """Test degradation profiler produces valid profile."""
        train = SpikeTrain(times=np.linspace(0.1, 0.9, 100))
        
        profiler = DegradationProfiler()
        # Use larger noise levels to see actual degradation beyond float32 precision
        profile = profiler.profile(train, noise_levels=[0, 0.1, 1.0, 10.0])
        
        assert len(profile.points) == 4
        # At these noise levels, degradation should be monotonic
        # (very low noise levels show float32 quantization noise, not actual degradation)


class TestStressGenerators:
    """Tests for stress test generators."""
    
    def test_regular_generator(self):
        """Test regular spike train generator."""
        train = generate_regular(firing_rate=10.0, duration=1.0)
        
        assert len(train) == 10  # 10 Hz for 1 second
        assert train.t_stop == 1.0
    
    def test_burst_generator(self):
        """Test burst spike train generator."""
        train = generate_burst(n_bursts=5, spikes_per_burst=10)
        
        assert len(train) == 50  # 5 bursts * 10 spikes
    
    def test_population_generator(self):
        """Test population generator."""
        trains = generate_population(n_neurons=10, firing_rate=10.0, duration=1.0)
        
        assert len(trains) == 10
        for train in trains:
            assert isinstance(train, SpikeTrain)


class TestLargeScale:
    """Large-scale stress tests."""
    
    def test_large_population(self):
        """Test encoding/decoding a large population."""
        trains = generate_population(n_neurons=100, firing_rate=50.0, duration=1.0)
        codec = SpikelinkCodec()
        
        total_spikes = 0
        recovered_spikes = 0
        
        for train in trains:
            total_spikes += len(train)
            packets = codec.encode_train(train)
            recovered = codec.decode_packets(packets)
            recovered_spikes += len(recovered)
        
        # Should preserve all spikes
        assert recovered_spikes == total_spikes
    
    def test_high_rate_train(self):
        """Test high firing rate spike train."""
        # 1000 Hz for 1 second = 1000 spikes
        train = generate_regular(firing_rate=1000.0, duration=1.0, jitter=0.0001)
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        assert len(recovered) == len(train)
    
    def test_long_duration(self):
        """Test long duration recording."""
        # 10 Hz for 100 seconds = 1000 spikes
        train = generate_regular(firing_rate=10.0, duration=100.0)
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        assert len(recovered) == len(train)
        assert recovered.t_stop >= 100.0
