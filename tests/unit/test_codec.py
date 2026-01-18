"""Unit tests for SpikelinkCodec."""

import numpy as np
import pytest

from spikelink import SpikeTrain, SpikelinkCodec


class TestSpikelinkCodec:
    """Tests for the SpikelinkCodec class."""
    
    def test_encode_decode_basic(self):
        """Test basic encode-decode round trip."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        codec = SpikelinkCodec()
        
        packets = codec.encode_train(train)
        recovered = codec.decode_packets(packets)
        
        assert len(recovered) == len(train)
        np.testing.assert_array_almost_equal(recovered.times, train.times, decimal=5)
    
    def test_encode_decode_empty(self):
        """Test round trip with empty spike train."""
        train = SpikeTrain(times=[])
        codec = SpikelinkCodec()
        
        packets = codec.encode_train(train)
        recovered = codec.decode_packets(packets)
        
        assert len(recovered) == 0
    
    def test_encode_decode_single_spike(self):
        """Test round trip with single spike."""
        train = SpikeTrain(times=[0.5])
        codec = SpikelinkCodec()
        
        packets = codec.encode_train(train)
        recovered = codec.decode_packets(packets)
        
        assert len(recovered) == 1
        np.testing.assert_almost_equal(recovered.times[0], 0.5, decimal=5)
    
    def test_encode_produces_packets(self):
        """Test that encoding produces packets."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3])
        codec = SpikelinkCodec()
        
        packets = codec.encode_train(train)
        
        assert len(packets) >= 1
        assert all(hasattr(p, 'times') for p in packets)
    
    def test_packet_chunking(self):
        """Test that large trains are split into multiple packets."""
        times = np.linspace(0, 10, 3000)  # More than default max_spikes_per_packet
        train = SpikeTrain(times=times)
        codec = SpikelinkCodec(max_spikes_per_packet=1024)
        
        packets = codec.encode_train(train)
        
        assert len(packets) > 1
        
        # Verify all spikes are preserved
        recovered = codec.decode_packets(packets)
        assert len(recovered) == len(train)
    
    def test_convenience_encode_times(self):
        """Test convenience method for encoding raw times."""
        codec = SpikelinkCodec()
        
        packets = codec.encode_times([0.1, 0.2, 0.3])
        times = codec.decode_times(packets)
        
        np.testing.assert_array_almost_equal(times, [0.1, 0.2, 0.3], decimal=5)
    
    def test_round_trip_method(self):
        """Test the round_trip convenience method."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        assert len(recovered) == len(train)
        np.testing.assert_array_almost_equal(recovered.times, train.times, decimal=5)
    
    def test_preserves_neuron_id(self):
        """Test that neuron ID is preserved through round trip."""
        train = SpikeTrain(times=[0.1, 0.2, 0.3], neuron_id=42)
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        assert recovered.neuron_id == 42


class TestSpikelinkCodecPrecision:
    """Tests for precision preservation."""
    
    def test_float32_precision(self):
        """Test precision limits of float32 transport."""
        # float32 has ~7 decimal digits of precision
        train = SpikeTrain(times=[0.1234567, 0.2345678, 0.3456789])
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        # Should preserve at least 5 decimals
        np.testing.assert_array_almost_equal(recovered.times, train.times, decimal=5)
    
    def test_small_intervals(self):
        """Test handling of small inter-spike intervals."""
        times = [0.1, 0.1001, 0.1002, 0.1003]  # 100 microsecond intervals
        train = SpikeTrain(times=times)
        codec = SpikelinkCodec()
        
        recovered = codec.round_trip(train)
        
        assert len(recovered) == len(train)
        # Intervals should be distinguishable
        isi = np.diff(recovered.times)
        assert np.all(isi > 0)
