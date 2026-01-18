"""Unit tests for SpikelinkPacket."""

import numpy as np
import pytest

from spikelink.core.packet import SpikelinkPacket, MAGIC, VERSION, HEADER_SIZE


class TestSpikelinkPacket:
    """Tests for the SpikelinkPacket class."""
    
    def test_create_basic(self):
        """Test basic packet creation."""
        times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        assert len(packet) == 3
        assert packet.t_start == 0.0
        assert packet.t_stop == 1.0
    
    def test_create_with_bounds(self):
        """Test packet creation with custom time bounds."""
        times = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        packet = SpikelinkPacket(times=times, t_start=0.0, t_stop=2.0)
        
        assert packet.t_start == 0.0
        assert packet.t_stop == 2.0
    
    def test_times_converted_to_float32(self):
        """Test that times are converted to float32."""
        times = [0.1, 0.2, 0.3]  # Python list
        packet = SpikelinkPacket(times=times)
        
        assert packet.times.dtype == np.float32
    
    def test_payload_size(self):
        """Test payload size calculation."""
        times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        assert packet.payload_size == 12  # 3 * 4 bytes
    
    def test_total_size(self):
        """Test total size calculation."""
        times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        assert packet.total_size == HEADER_SIZE + 12


class TestSpikelinkPacketSerialization:
    """Tests for packet serialization."""
    
    def test_to_bytes_header(self):
        """Test that serialization produces correct header."""
        times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        data = packet.to_bytes()
        
        assert data[:4] == MAGIC
        assert data[4] == VERSION
    
    def test_round_trip_serialization(self):
        """Test serialize-deserialize round trip."""
        times = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        original = SpikelinkPacket(times=times, t_start=0.0, t_stop=1.0)
        
        data = original.to_bytes()
        recovered = SpikelinkPacket.from_bytes(data)
        
        assert len(recovered) == len(original)
        np.testing.assert_array_almost_equal(recovered.times, original.times)
        assert recovered.t_start == original.t_start
        assert recovered.t_stop == original.t_stop
    
    def test_from_bytes_invalid_magic(self):
        """Test that invalid magic number raises error."""
        data = b"XXXX" + b"\x00" * 20
        
        with pytest.raises(ValueError, match="Invalid magic"):
            SpikelinkPacket.from_bytes(data)
    
    def test_from_bytes_truncated(self):
        """Test that truncated data raises error."""
        data = MAGIC + b"\x01"  # Too short
        
        with pytest.raises(ValueError, match="too short"):
            SpikelinkPacket.from_bytes(data)
    
    def test_empty_packet_serialization(self):
        """Test serialization of empty packet."""
        packet = SpikelinkPacket(times=np.array([], dtype=np.float32))
        
        data = packet.to_bytes()
        recovered = SpikelinkPacket.from_bytes(data)
        
        assert len(recovered) == 0


class TestSpikelinkPacketIntegrity:
    """Tests for packet integrity verification."""
    
    def test_verify_valid_packet(self):
        """Test that valid packet passes integrity check."""
        times = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times, t_start=0.0, t_stop=1.0)
        
        assert packet.verify_integrity() is True
    
    def test_verify_times_out_of_bounds(self):
        """Test that out-of-bounds times fail integrity check."""
        times = np.array([0.1, 0.2, 1.5], dtype=np.float32)  # 1.5 > t_stop
        packet = SpikelinkPacket(times=times, t_start=0.0, t_stop=1.0)
        
        assert packet.verify_integrity() is False
    
    def test_verify_nan_times(self):
        """Test that NaN times fail integrity check."""
        times = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        assert packet.verify_integrity() is False
    
    def test_verify_inf_times(self):
        """Test that infinite times fail integrity check."""
        times = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        packet = SpikelinkPacket(times=times)
        
        assert packet.verify_integrity() is False
