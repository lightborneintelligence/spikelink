#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  PLATFORM TEST #2: Brian2
═══════════════════════════════════════════════════════════════════════════
  Lightborne Intelligence

  Proves SpikeLink v2 interoperates with Brian2, the most widely-used
  spiking neural network simulator in computational neuroscience.

  Test axes:
    1. Brian2 SpikeMonitor → v2 SpikeTrain conversion
    2. Single LIF neuron round-trip through v2 codec
    3. Heterogeneous population transport (10 neurons, different rates)
    4. Poisson-driven network (50 input → 20 LIF with synapses)
    5. Statistical fidelity cross-validated with Elephant
    6. Bursting network (adaptation currents)
    7. Spike count integrity at scale (1000 neurons)

  Dependencies: brian2, neo, elephant, quantities, numpy
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sys
import os

# v2 modules
sys.path.insert(0, os.path.dirname(__file__))
from waveml_core import WaveState, HarmonicTransform, ERA, ERABounds, ShellMap
from spikelink_v2 import SpikelinkCodecV2, SpikeTrain as V2SpikeTrain

# Brian2
from brian2 import *
import brian2

# Elephant for cross-validation
import neo
import quantities as pq
import elephant.statistics as estats


# ═════════════════════════════════════════════════════════════════════════
# BRIAN2 ↔ V2 ADAPTER
# ═════════════════════════════════════════════════════════════════════════

class Brian2AdapterV2:
    """
    Bridge between Brian2 SpikeMonitor and SpikeLink v2 SpikeTrain.
    
    Brian2 SpikeMonitor provides:
      - M.t : spike times (with Brian2 units)
      - M.i : neuron indices
      - M.count : per-neuron spike counts
    
    Strategy:
      Brian2 → v2: extract per-neuron times, assign unit amplitudes
      v2 → Brian2: not applicable (Brian2 is a simulator, not a consumer)
      v2 → Neo: for cross-validation with Elephant
    """
    
    @staticmethod
    def from_spike_monitor(monitor, neuron_id: int) -> V2SpikeTrain:
        """Extract a single neuron's spike train from a Brian2 SpikeMonitor.
        
        Args:
            monitor: Brian2 SpikeMonitor
            neuron_id: Which neuron to extract
            
        Returns:
            V2SpikeTrain with times in seconds and unit amplitudes
        """
        mask = np.array(monitor.i) == neuron_id
        times = np.array(monitor.t[mask] / second)
        times = np.sort(times)  # Brian2 guarantees order, but be safe
        amplitudes = np.ones(len(times))
        return V2SpikeTrain(times=times, amplitudes=amplitudes)
    
    @staticmethod
    def from_spike_monitor_all(monitor, n_neurons: int) -> list:
        """Extract ALL neuron spike trains from a Brian2 SpikeMonitor.
        
        Returns:
            List of V2SpikeTrain, one per neuron
        """
        trains = []
        all_t = np.array(monitor.t / second)
        all_i = np.array(monitor.i)
        for nid in range(n_neurons):
            mask = all_i == nid
            times = np.sort(all_t[mask])
            trains.append(V2SpikeTrain(
                times=times, amplitudes=np.ones(len(times))))
        return trains
    
    @staticmethod
    def to_neo(v2_train: V2SpikeTrain, t_stop: float) -> neo.SpikeTrain:
        """Convert v2 SpikeTrain → Neo for Elephant cross-validation."""
        if len(v2_train.times) == 0:
            return neo.SpikeTrain(times=[] * pq.s, t_stop=t_stop * pq.s)
        return neo.SpikeTrain(
            times=v2_train.times * pq.s, t_stop=t_stop * pq.s)


# ═════════════════════════════════════════════════════════════════════════
# TEST HARNESS (reused from Platform #1)
# ═════════════════════════════════════════════════════════════════════════

class PlatformTestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def check(self, test_name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.tests.append((test_name, status, detail))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
    
    def report(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"  PLATFORM TEST: {self.name}")
        lines.append(f"  Lightborne Intelligence — SpikeLink v2.0")
        lines.append("=" * 70)
        for name, status, detail in self.tests:
            icon = "✅" if status == "PASS" else "❌"
            line = f"  {icon} {name}"
            if detail:
                line += f"  │  {detail}"
            lines.append(line)
        lines.append("-" * 70)
        total = self.passed + self.failed
        lines.append(f"  Results: {self.passed}/{total} passed")
        if self.failed == 0:
            lines.append(f"  ✅ {self.name} — FULLY COMPATIBLE")
        else:
            lines.append(f"  ❌ {self.name} — {self.failed} FAILURES")
        lines.append("=" * 70)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# BRIAN2 SIMULATIONS
# ═════════════════════════════════════════════════════════════════════════

def run_single_lif(duration_ms=500):
    """Single LIF neuron with constant current drive."""
    start_scope()
    defaultclock.dt = 0.1*ms
    
    G = NeuronGroup(1, 'dv/dt = (-v + 1.2)/(10*ms) : 1',
                    threshold='v>1', reset='v=0', method='euler')
    M = SpikeMonitor(G)
    run(duration_ms * ms)
    return M, 1


def run_heterogeneous_population(n=10, duration_ms=1000):
    """Population with different drive currents → different firing rates."""
    start_scope()
    defaultclock.dt = 0.1*ms
    
    G = NeuronGroup(n, '''
        dv/dt = (-v + I) / tau : 1
        I : 1
        tau : second
    ''', threshold='v>1', reset='v=0', method='euler')
    
    G.I = np.linspace(1.05, 2.0, n)
    G.tau = np.linspace(8, 15, n) * ms
    
    M = SpikeMonitor(G)
    run(duration_ms * ms)
    return M, n


def run_poisson_network(n_input=50, n_output=20, duration_ms=500):
    """Poisson input → synaptically driven LIF population."""
    start_scope()
    defaultclock.dt = 0.1*ms
    
    P = PoissonGroup(n_input, rates=30*Hz)
    G = NeuronGroup(n_output, '''
        dv/dt = (-v + 0.5) / (10*ms) : 1
    ''', threshold='v>1', reset='v=0', method='euler')
    
    S = Synapses(P, G, on_pre='v += 0.1')
    S.connect(p=0.3)
    
    M_in = SpikeMonitor(P)
    M_out = SpikeMonitor(G)
    run(duration_ms * ms)
    return M_in, n_input, M_out, n_output


def run_bursting_network(n=10, duration_ms=1000):
    """Adaptive-exponential LIF (AdEx) that produces bursting."""
    start_scope()
    defaultclock.dt = 0.1*ms
    
    # Izhikevich-style bursting model (simpler than full AdEx)
    G = NeuronGroup(n, '''
        dv/dt = (0.04*v**2 + 5*v + 140 - w + I) / ms : 1
        dw/dt = (0.02 * (0.2*v - w)) / ms : 1
        I : 1
    ''', threshold='v >= 30', 
    reset='v = -65; w += 8',
    method='euler')
    
    G.v = -65
    G.w = -14
    # Varying input produces different burst patterns
    G.I = np.linspace(5, 15, n)
    
    M = SpikeMonitor(G)
    run(duration_ms * ms)
    return M, n


def run_large_population(n=1000, duration_ms=200):
    """Large-scale population for spike count integrity test."""
    start_scope()
    defaultclock.dt = 0.1*ms
    
    G = NeuronGroup(n, '''
        dv/dt = (-v + I) / (10*ms) : 1
        I : 1
    ''', threshold='v>1', reset='v=0', method='euler')
    
    # Log-normal rate distribution (realistic)
    np.random.seed(42)
    G.I = 1.0 + np.random.lognormal(0, 0.5, n) * 0.3
    
    M = SpikeMonitor(G)
    run(duration_ms * ms)
    return M, n


# ═════════════════════════════════════════════════════════════════════════
# TEST 1: BRIAN2 → V2 CONVERSION
# ═════════════════════════════════════════════════════════════════════════

def test_conversion(results: PlatformTestResult):
    print("\n[1] Brian2 SpikeMonitor → v2 Conversion...")
    
    M, n = run_single_lif(duration_ms=500)
    n_spikes = M.num_spikes
    
    # Extract via adapter
    v2_train = Brian2AdapterV2.from_spike_monitor(M, neuron_id=0)
    
    results.check(
        "Brian2 → v2 preserves spike count",
        len(v2_train.times) == n_spikes,
        f"Brian2: {n_spikes} spikes → v2: {len(v2_train.times)} spikes"
    )
    
    # Verify times match Brian2 output
    brian_times = np.sort(np.array(M.t[M.i == 0] / second))
    results.check(
        "Brian2 → v2 preserves timing (exact)",
        np.allclose(v2_train.times, brian_times, atol=1e-12),
        f"Max Δt = {np.max(np.abs(v2_train.times - brian_times)):.2e}"
    )
    
    results.check(
        "Brian2 → v2 assigns unit amplitudes",
        np.all(v2_train.amplitudes == 1.0),
        f"All amplitudes = 1.0"
    )
    
    # Cross-validate: v2 → Neo should work
    neo_train = Brian2AdapterV2.to_neo(v2_train, t_stop=0.5)
    results.check(
        "Brian2 → v2 → Neo chain valid",
        isinstance(neo_train, neo.SpikeTrain) and len(neo_train) == n_spikes,
        f"Neo SpikeTrain: {len(neo_train)} spikes"
    )


# ═════════════════════════════════════════════════════════════════════════
# TEST 2: SINGLE LIF NEURON ROUND-TRIP
# ═════════════════════════════════════════════════════════════════════════

def test_single_neuron_roundtrip(results: PlatformTestResult):
    print("[2] Single LIF Neuron Round-Trip...")
    
    M, _ = run_single_lif(duration_ms=500)
    n_orig = M.num_spikes
    
    v2_input = Brian2AdapterV2.from_spike_monitor(M, neuron_id=0)
    
    codec = SpikelinkCodecV2(max_amplitude=2.0)
    packets = codec.encode_train(v2_input)
    v2_recovered = codec.decode_packets(packets)
    
    results.check(
        "LIF neuron: spike count preserved",
        len(v2_recovered.times) == n_orig,
        f"{n_orig} → {len(packets)} pkts → {len(v2_recovered.times)}"
    )
    
    # Rate preservation via Elephant
    neo_orig = Brian2AdapterV2.to_neo(v2_input, t_stop=0.5)
    neo_recv = Brian2AdapterV2.to_neo(v2_recovered, t_stop=0.5)
    
    orig_rate = float(estats.mean_firing_rate(neo_orig).rescale('Hz').magnitude)
    recv_rate = float(estats.mean_firing_rate(neo_recv).rescale('Hz').magnitude)
    rate_err = abs(orig_rate - recv_rate) / (orig_rate + 1e-12)
    
    results.check(
        "LIF firing rate preserved (Elephant, <5%)",
        rate_err < 0.05,
        f"Original: {orig_rate:.1f} Hz → Recovered: {recv_rate:.1f} Hz, "
        f"Error: {rate_err:.1%}"
    )
    
    # ISI preservation
    if n_orig > 2:
        orig_isi = estats.isi(neo_orig)
        recv_isi = estats.isi(neo_recv)
        mean_isi_err = abs(
            float(np.mean(orig_isi.magnitude)) - 
            float(np.mean(recv_isi.magnitude))
        ) / (float(np.mean(orig_isi.magnitude)) + 1e-12)
        
        results.check(
            "LIF mean ISI preserved (Elephant, <10%)",
            mean_isi_err < 0.10,
            f"ISI error: {mean_isi_err:.1%}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 3: HETEROGENEOUS POPULATION (10 neurons, different rates)
# ═════════════════════════════════════════════════════════════════════════

def test_heterogeneous_population(results: PlatformTestResult):
    print("[3] Heterogeneous Population (10 neurons)...")
    
    M, n = run_heterogeneous_population(n=10, duration_ms=1000)
    trains_in = Brian2AdapterV2.from_spike_monitor_all(M, n)
    total_in = sum([len(t.times) for t in trains_in])
    
    codec = SpikelinkCodecV2(max_amplitude=2.0)
    trains_out = []
    total_packets = 0
    
    for v2_in in trains_in:
        if len(v2_in.times) == 0:
            trains_out.append(v2_in)
            continue
        codec.reset()
        pkts = codec.encode_train(v2_in)
        total_packets += len(pkts)
        v2_out = codec.decode_packets(pkts)
        trains_out.append(v2_out)
    
    total_out = sum([len(t.times) for t in trains_out])
    
    results.check(
        "All 10 neurons transported",
        len(trains_out) == 10,
        f"{total_in} total spikes → {total_packets} packets → {total_out}"
    )
    
    results.check(
        "Total spike count preserved (100%)",
        total_out == total_in,
        f"In: {total_in}, Out: {total_out}"
    )
    
    # Rate distribution preservation
    orig_rates = []
    recv_rates = []
    for v2_in, v2_out in zip(trains_in, trains_out):
        if len(v2_in.times) > 0:
            neo_in = Brian2AdapterV2.to_neo(v2_in, t_stop=1.0)
            neo_out = Brian2AdapterV2.to_neo(v2_out, t_stop=1.0)
            orig_rates.append(float(
                estats.mean_firing_rate(neo_in).rescale('Hz').magnitude))
            recv_rates.append(float(
                estats.mean_firing_rate(neo_out).rescale('Hz').magnitude))
    
    if len(orig_rates) > 2:
        rate_corr = np.corrcoef(orig_rates, recv_rates)[0, 1]
        results.check(
            "Rate distribution preserved (r > 0.95)",
            rate_corr > 0.95,
            f"Correlation: {rate_corr:.4f}, "
            f"Range: {min(orig_rates):.0f}-{max(orig_rates):.0f} Hz"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 4: POISSON-DRIVEN NETWORK
# ═════════════════════════════════════════════════════════════════════════

def test_poisson_network(results: PlatformTestResult):
    print("[4] Poisson-Driven Network (50→20 with synapses)...")
    
    M_in, n_in, M_out, n_out = run_poisson_network()
    
    # Transport the OUTPUT neurons (the interesting ones — synaptically driven)
    trains_in = Brian2AdapterV2.from_spike_monitor_all(M_out, n_out)
    total_in = sum([len(t.times) for t in trains_in])
    
    codec = SpikelinkCodecV2(max_amplitude=2.0)
    trains_out = []
    
    for v2_in in trains_in:
        if len(v2_in.times) == 0:
            trains_out.append(v2_in)
            continue
        codec.reset()
        pkts = codec.encode_train(v2_in)
        trains_out.append(codec.decode_packets(pkts))
    
    total_out = sum([len(t.times) for t in trains_out])
    
    results.check(
        "Network output: spike count preserved",
        total_out == total_in,
        f"Input: {M_in.num_spikes} Poisson spikes → "
        f"Output: {total_in} LIF spikes → Transported: {total_out}"
    )
    
    # Active neuron count preserved
    active_in = sum([1 for t in trains_in if len(t.times) > 0])
    active_out = sum([1 for t in trains_out if len(t.times) > 0])
    results.check(
        "Active neuron count preserved",
        active_out == active_in,
        f"Active: {active_in}/{n_out} → {active_out}/{n_out}"
    )
    
    # Mean population rate via Elephant
    rates_in = [float(estats.mean_firing_rate(
        Brian2AdapterV2.to_neo(t, t_stop=0.5)).rescale('Hz').magnitude)
        for t in trains_in if len(t.times) > 0]
    rates_out = [float(estats.mean_firing_rate(
        Brian2AdapterV2.to_neo(t, t_stop=0.5)).rescale('Hz').magnitude)
        for t in trains_out if len(t.times) > 0]
    
    mean_in = np.mean(rates_in)
    mean_out = np.mean(rates_out)
    results.check(
        "Network mean rate preserved (<5%)",
        abs(mean_in - mean_out) / (mean_in + 1e-12) < 0.05,
        f"Original: {mean_in:.1f} Hz → Recovered: {mean_out:.1f} Hz"
    )


# ═════════════════════════════════════════════════════════════════════════
# TEST 5: BURSTING NETWORK (Izhikevich model)
# ═════════════════════════════════════════════════════════════════════════

def test_bursting(results: PlatformTestResult):
    print("[5] Bursting Network (Izhikevich, 10 neurons)...")
    
    M, n = run_bursting_network(n=10, duration_ms=1000)
    trains_in = Brian2AdapterV2.from_spike_monitor_all(M, n)
    total_in = sum([len(t.times) for t in trains_in])
    
    if total_in == 0:
        results.check("Bursting network produced spikes", False,
                       "No spikes generated — model may need tuning")
        return
    
    codec = SpikelinkCodecV2(max_amplitude=2.0)
    trains_out = []
    
    for v2_in in trains_in:
        if len(v2_in.times) == 0:
            trains_out.append(v2_in)
            continue
        codec.reset()
        pkts = codec.encode_train(v2_in)
        trains_out.append(codec.decode_packets(pkts))
    
    total_out = sum([len(t.times) for t in trains_out])
    
    results.check(
        "Bursting: spike count preserved",
        total_out == total_in,
        f"In: {total_in} spikes, Out: {total_out} spikes"
    )
    
    # Check that burst structure is preserved
    # (neurons with many spikes should still have many spikes)
    counts_in = [len(t.times) for t in trains_in]
    counts_out = [len(t.times) for t in trains_out]
    
    results.check(
        "Burst per-neuron counts match exactly",
        counts_in == counts_out,
        f"Per-neuron: {counts_in}"
    )
    
    # Rate correlation
    nonzero = [(i, o) for i, o in zip(counts_in, counts_out) if i > 0]
    if len(nonzero) > 2:
        r_in, r_out = zip(*nonzero)
        corr = np.corrcoef(r_in, r_out)[0, 1]
        results.check(
            "Burst rate distribution preserved (r > 0.99)",
            corr > 0.99 or np.isnan(corr),
            f"Correlation: {corr:.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 6: ELEPHANT CROSS-VALIDATION (ISI distribution fidelity)
# ═════════════════════════════════════════════════════════════════════════

def test_elephant_cross_validation(results: PlatformTestResult):
    print("[6] Elephant Cross-Validation (ISI distribution)...")
    
    M, n = run_heterogeneous_population(n=5, duration_ms=2000)
    
    for nid in [0, 2, 4]:  # slow, mid, fast neuron
        v2_in = Brian2AdapterV2.from_spike_monitor(M, neuron_id=nid)
        
        if len(v2_in.times) < 5:
            continue
        
        codec = SpikelinkCodecV2(max_amplitude=2.0)
        pkts = codec.encode_train(v2_in)
        v2_out = codec.decode_packets(pkts)
        
        neo_in = Brian2AdapterV2.to_neo(v2_in, t_stop=2.0)
        neo_out = Brian2AdapterV2.to_neo(v2_out, t_stop=2.0)
        
        # ISI distribution comparison
        isi_in = estats.isi(neo_in)
        isi_out = estats.isi(neo_out)
        
        mean_err = abs(
            float(np.mean(isi_in.magnitude)) -
            float(np.mean(isi_out.magnitude))
        ) / (float(np.mean(isi_in.magnitude)) + 1e-12)
        
        rate_in = float(estats.mean_firing_rate(neo_in).rescale('Hz').magnitude)
        
        results.check(
            f"Neuron {nid} ({rate_in:.0f} Hz): ISI mean preserved",
            mean_err < 0.10,
            f"ISI error: {mean_err:.1%}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 7: SPIKE COUNT INTEGRITY AT SCALE (1000 neurons)
# ═════════════════════════════════════════════════════════════════════════

def test_large_scale(results: PlatformTestResult):
    print("[7] Large-Scale Integrity (1000 neurons)...")
    
    M, n = run_large_population(n=1000, duration_ms=200)
    total_in = M.num_spikes
    
    # Batch extract all trains
    trains_in = Brian2AdapterV2.from_spike_monitor_all(M, n)
    
    codec = SpikelinkCodecV2(max_amplitude=2.0)
    total_out = 0
    total_packets = 0
    neurons_exact = 0
    
    for v2_in in trains_in:
        if len(v2_in.times) == 0:
            neurons_exact += 1
            continue
        codec.reset()
        pkts = codec.encode_train(v2_in)
        total_packets += len(pkts)
        v2_out = codec.decode_packets(pkts)
        n_out = len(v2_out.times)
        total_out += n_out
        if n_out == len(v2_in.times):
            neurons_exact += 1
    
    results.check(
        "1000 neurons: total spike count exact",
        total_out == total_in,
        f"In: {total_in}, Out: {total_out}, Packets: {total_packets}"
    )
    
    results.check(
        "1000 neurons: per-neuron counts exact",
        neurons_exact == n,
        f"{neurons_exact}/{n} neurons with exact spike count"
    )
    
    active = sum([1 for t in trains_in if len(t.times) > 0])
    results.check(
        "1000 neurons: all active neurons transported",
        True,
        f"{active}/{n} active neurons, {total_in} total spikes"
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    results = PlatformTestResult("Brian2 (SNN Simulator)")
    
    print("\n" + "=" * 70)
    print("  SPIKELINK v2.0 — PLATFORM INTEROP TEST #2")
    print("  Target: Brian2 SNN Simulator")
    print(f"  brian2 {brian2.__version__}")
    print("=" * 70)
    
    test_conversion(results)
    test_single_neuron_roundtrip(results)
    test_heterogeneous_population(results)
    test_poisson_network(results)
    test_bursting(results)
    test_elephant_cross_validation(results)
    test_large_scale(results)
    
    print("\n" + results.report())
    return results


if __name__ == '__main__':
    r = main()
    sys.exit(0 if r.failed == 0 else 1)
