"""
Stress test generators — Create spike trains for stress testing.

Generates various spike train patterns for testing SpikeLink
under realistic neuromorphic workloads.
"""

from typing import List, Optional
import numpy as np

from spikelink.types.spiketrain import SpikeTrain


def generate_regular(
    firing_rate: float = 10.0,
    duration: float = 1.0,
    jitter: float = 0.0,
    seed: Optional[int] = None,
) -> SpikeTrain:
    """
    Generate a regular (periodic) spike train.
    
    Args:
        firing_rate: Spikes per second (Hz)
        duration: Duration in seconds
        jitter: Standard deviation of timing jitter (seconds)
        seed: Random seed for jitter
        
    Returns:
        SpikeTrain with regular firing pattern
    """
    if seed is not None:
        np.random.seed(seed)
    
    isi = 1.0 / firing_rate
    n_spikes = int(duration * firing_rate)
    
    times = np.arange(n_spikes) * isi + isi / 2
    
    if jitter > 0:
        times += np.random.normal(0, jitter, n_spikes)
        times = np.clip(times, 0, duration)
        times = np.sort(times)
    
    return SpikeTrain(times=times, t_start=0.0, t_stop=duration)


def generate_burst(
    n_bursts: int = 5,
    spikes_per_burst: int = 10,
    burst_duration: float = 0.05,
    inter_burst_interval: float = 0.2,
    seed: Optional[int] = None,
) -> SpikeTrain:
    """
    Generate a bursting spike train.
    
    Args:
        n_bursts: Number of bursts
        spikes_per_burst: Spikes in each burst
        burst_duration: Duration of each burst (seconds)
        inter_burst_interval: Time between burst starts (seconds)
        seed: Random seed
        
    Returns:
        SpikeTrain with bursting pattern
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_times = []
    
    for i in range(n_bursts):
        burst_start = i * inter_burst_interval
        burst_times = burst_start + np.random.uniform(0, burst_duration, spikes_per_burst)
        all_times.extend(burst_times)
    
    times = np.sort(np.array(all_times))
    duration = n_bursts * inter_burst_interval
    
    return SpikeTrain(times=times, t_start=0.0, t_stop=duration)


def generate_population(
    n_neurons: int = 100,
    firing_rate: float = 10.0,
    duration: float = 1.0,
    correlation: float = 0.0,
    seed: Optional[int] = None,
) -> List[SpikeTrain]:
    """
    Generate a population of spike trains.
    
    Args:
        n_neurons: Number of neurons in population
        firing_rate: Mean firing rate (Hz)
        duration: Duration in seconds
        correlation: Pairwise correlation (0 = independent, 1 = identical)
        seed: Random seed
        
    Returns:
        List of SpikeTrains, one per neuron
    """
    if seed is not None:
        np.random.seed(seed)
    
    trains = []
    
    # Generate shared spikes for correlation
    if correlation > 0:
        shared_rate = firing_rate * correlation
        independent_rate = firing_rate * (1 - correlation)
        
        # Shared spike times
        n_shared = np.random.poisson(shared_rate * duration)
        shared_times = np.sort(np.random.uniform(0, duration, n_shared))
    else:
        independent_rate = firing_rate
        shared_times = np.array([])
    
    for i in range(n_neurons):
        # Independent spikes for this neuron
        n_independent = np.random.poisson(independent_rate * duration)
        independent_times = np.random.uniform(0, duration, n_independent)
        
        # Combine shared and independent
        if len(shared_times) > 0:
            all_times = np.concatenate([shared_times, independent_times])
        else:
            all_times = independent_times
        
        times = np.sort(all_times)
        
        train = SpikeTrain(
            times=times,
            t_start=0.0,
            t_stop=duration,
            neuron_id=i,
        )
        trains.append(train)
    
    return trains


def generate_poisson(
    firing_rate: float = 10.0,
    duration: float = 1.0,
    seed: Optional[int] = None,
) -> SpikeTrain:
    """
    Generate a Poisson spike train.
    
    Args:
        firing_rate: Mean firing rate (Hz)
        duration: Duration in seconds
        seed: Random seed
        
    Returns:
        SpikeTrain with Poisson statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate inter-spike intervals from exponential distribution
    mean_isi = 1.0 / firing_rate
    n_expected = int(duration * firing_rate * 2)  # Generate extra, then trim
    
    isis = np.random.exponential(mean_isi, n_expected)
    times = np.cumsum(isis)
    
    # Keep only spikes within duration
    times = times[times < duration]
    
    return SpikeTrain(times=times, t_start=0.0, t_stop=duration)
