"""
DegradationProfiler — Characterize graceful degradation under noise.

Measures how precision degrades as noise increases, verifying
that SpikeLink provides smooth degradation rather than catastrophic failure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from spikelink.types.spiketrain import SpikeTrain
from spikelink.core.codec import SpikelinkCodec


@dataclass
class DegradationPoint:
    """Single point on degradation curve."""
    noise_level: float
    mean_error: float
    max_error: float
    decimals_preserved: int
    spike_preservation_rate: float


@dataclass 
class DegradationProfile:
    """Complete degradation profile across noise levels."""
    points: List[DegradationPoint]
    is_monotonic: bool
    
    def summary(self) -> str:
        """Return summary string."""
        mono = "✓ Monotonic" if self.is_monotonic else "✗ Non-monotonic"
        return f"DegradationProfile: {len(self.points)} points, {mono}"


class DegradationProfiler:
    """
    Profile how SpikeLink degrades under increasing noise.
    
    Verifies the key property: precision loss under noise, not data loss.
    Degradation should be smooth and monotonic, not catastrophic.
    
    Example:
        >>> profiler = DegradationProfiler()
        >>> train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        >>> profile = profiler.profile(train)
        >>> profiler.print_profile(profile)
    """
    
    def __init__(self, codec: Optional[SpikelinkCodec] = None):
        """
        Initialize profiler.
        
        Args:
            codec: Codec to test (default: standard SpikelinkCodec)
        """
        self.codec = codec or SpikelinkCodec()
    
    def add_noise(
        self,
        train: SpikeTrain,
        noise_level: float,
        seed: Optional[int] = None,
    ) -> SpikeTrain:
        """
        Add Gaussian noise to spike times.
        
        Args:
            train: Original spike train
            noise_level: Standard deviation of noise (as fraction of mean ISI)
            seed: Random seed for reproducibility
            
        Returns:
            Noisy spike train
        """
        if seed is not None:
            np.random.seed(seed)
        
        if len(train) == 0 or noise_level == 0:
            return train.copy()
        
        # Scale noise by mean ISI (or duration if only 1 spike)
        if len(train) > 1:
            scale = np.mean(np.diff(train.times))
        else:
            scale = train.duration
        
        noise = np.random.normal(0, noise_level * scale, len(train.times))
        noisy_times = train.times + noise
        
        # Clip to valid range
        noisy_times = np.clip(noisy_times, train.t_start, train.t_stop)
        
        # Re-sort (noise may disorder spikes)
        noisy_times = np.sort(noisy_times)
        
        return SpikeTrain(
            times=noisy_times,
            t_start=train.t_start,
            t_stop=train.t_stop,
            units=train.units,
            neuron_id=train.neuron_id,
        )
    
    def measure_degradation(
        self,
        original: SpikeTrain,
        noisy: SpikeTrain,
    ) -> DegradationPoint:
        """
        Measure degradation between original and noisy trains.
        
        Args:
            original: Original spike train
            noisy: Noisy version after transport
            
        Returns:
            DegradationPoint with metrics
        """
        # Round-trip the noisy train
        recovered = self.codec.round_trip(noisy)
        
        # Compare recovered to the noisy input (not original)
        # This measures transport fidelity, not noise removal
        if len(noisy) == 0 or len(recovered) == 0:
            return DegradationPoint(
                noise_level=0.0,
                mean_error=0.0,
                max_error=0.0,
                decimals_preserved=6,
                spike_preservation_rate=1.0 if len(noisy) == len(recovered) else 0.0,
            )
        
        # Compute errors
        min_len = min(len(noisy), len(recovered))
        errors = np.abs(noisy.times[:min_len] - recovered.times[:min_len])
        
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        
        # Estimate preserved decimals
        if max_error == 0:
            decimals = 6
        else:
            decimals = max(0, int(-np.log10(max_error)))
        
        # Spike preservation rate
        preservation = len(recovered) / len(noisy) if len(noisy) > 0 else 1.0
        
        return DegradationPoint(
            noise_level=0.0,  # Will be set by profile()
            mean_error=mean_error,
            max_error=max_error,
            decimals_preserved=decimals,
            spike_preservation_rate=preservation,
        )
    
    def profile(
        self,
        train: SpikeTrain,
        noise_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> DegradationProfile:
        """
        Profile degradation across noise levels.
        
        Args:
            train: Original spike train
            noise_levels: List of noise levels to test (default: [0, 0.001, 0.01, 0.1, 1.0])
            seed: Random seed for reproducibility
            
        Returns:
            DegradationProfile
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.001, 0.01, 0.1, 1.0]
        
        points = []
        
        for i, noise_level in enumerate(noise_levels):
            # Add noise
            noisy = self.add_noise(train, noise_level, seed=seed + i)
            
            # Measure degradation
            point = self.measure_degradation(train, noisy)
            point.noise_level = noise_level
            
            points.append(point)
        
        # Check monotonicity (errors should increase with noise)
        errors = [p.max_error for p in points]
        is_monotonic = all(errors[i] <= errors[i + 1] + 1e-9 for i in range(len(errors) - 1))
        
        return DegradationProfile(points=points, is_monotonic=is_monotonic)
    
    def print_profile(self, profile: DegradationProfile):
        """
        Print degradation profile in readable format.
        
        Args:
            profile: Profile to print
        """
        print("=" * 60)
        print("SpikeLink Degradation Profile")
        print("=" * 60)
        print(f"{'Noise %':<10} {'Mean Err':<12} {'Max Err':<12} {'Decimals':<10} {'Preserved':<10}")
        print("-" * 60)
        
        for p in profile.points:
            print(
                f"{p.noise_level * 100:>6.2f}%    "
                f"{p.mean_error:<12.2e} "
                f"{p.max_error:<12.2e} "
                f"{p.decimals_preserved:<10} "
                f"{p.spike_preservation_rate * 100:>6.1f}%"
            )
        
        print("-" * 60)
        mono_status = "✓ Monotonic degradation confirmed" if profile.is_monotonic else "✗ Non-monotonic degradation detected"
        print(mono_status)
        print("=" * 60)
