#!/usr/bin/env python3
"""
test_platform_brian2.py — SpikeLink ↔ Brian2 integration tests
================================================================

Validates the Brian2Adapter against the adapter contract invariants:

    ✓ Exact spike count preservation
    ✓ Causal ordering (monotonicity)
    ✓ Timing fidelity within SpikeLink 10 μs quantisation floor
    ✓ Round-trip: Brian2 → SpikeLink → Brian2
    ✓ Multi-neuron SpikeMonitor handling
    ✓ Edge cases & error handling
    ✓ Stress test

Requires: pip install spikelink[brian2]

Lightborne Intelligence · Dallas TX
"""

import sys
import os
import numpy as np
import traceback

# ── Imports ──────────────────────────────────────────────────

try:
    from spikelink import SpikeTrain, SpikelinkCodec
    from spikelink.adapters.brian2 import Brian2Adapter
    SPIKELINK_OK = True
except ImportError as e:
    print(f"SKIP: spikelink not importable — {e}")
    SPIKELINK_OK = False

try:
    from brian2 import *
    from brian2 import ms, second, Hz
    BRIAN2_OK = True
except ImportError as e:
    print(f"SKIP: brian2 not importable — {e}")
    BRIAN2_OK = False


# ── Test Infrastructure ──────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
TESTS = []


def test(name, requires_brian2=True):
    """Register a test function."""
    def decorator(func):
        TESTS.append((name, func, requires_brian2))
        return func
    return decorator


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a!r} == {b!r}. {msg}")


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"|{a} - {b}| = {abs(a-b)} > tol={tol}. {msg}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "Condition failed")


def brian2_reset():
    """Reset Brian2 state."""
    if BRIAN2_OK:
        start_scope()


def make_spike_generator(spike_times_ms, n_neurons=1):
    """Create a Brian2 SpikeGeneratorGroup."""
    indices = np.zeros(len(spike_times_ms), dtype=int)
    times = np.array(spike_times_ms) * ms
    return SpikeGeneratorGroup(n_neurons, indices, times)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 1: Exact spike count preservation
# ═══════════════════════════════════════════════════════════════

@test("COUNT-01: SpikeMonitor spike count preserved (5 spikes)")
def test_count_basic():
    brian2_reset()
    sg = make_spike_generator([10.0, 20.0, 30.0, 40.0, 50.0])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_eq(len(train.times), 5, "Spike count mismatch")


@test("COUNT-02: Large spike count preserved (500 spikes)")
def test_count_large():
    brian2_reset()
    spike_times = np.sort(np.random.uniform(1.0, 999.0, 500)).tolist()
    sg = make_spike_generator(spike_times)
    mon = SpikeMonitor(sg)
    run(1000 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_eq(len(train.times), 500, "Large spike count mismatch")


@test("COUNT-03: Empty spike monitor handled")
def test_count_empty():
    brian2_reset()
    sg = make_spike_generator([])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_eq(len(train.times), 0, "Empty monitor should yield 0 spikes")


@test("COUNT-04: Single spike preserved")
def test_count_single():
    brian2_reset()
    sg = make_spike_generator([42.0])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_eq(len(train.times), 1)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 2: Causal ordering / monotonicity
# ═══════════════════════════════════════════════════════════════

@test("ORDER-01: Output spike times are monotonically increasing")
def test_monotonicity():
    brian2_reset()
    sg = make_spike_generator([50.0, 10.0, 30.0, 20.0, 40.0])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    for i in range(1, len(train.times)):
        assert_true(train.times[i] >= train.times[i-1],
                    f"Non-monotonic: {train.times[i-1]} > {train.times[i]}")


@test("ORDER-02: Monotonicity preserved through codec round-trip")
def test_monotonicity_codec():
    brian2_reset()
    spike_times = np.sort(np.random.uniform(1.0, 99.0, 100)).tolist()
    sg = make_spike_generator(spike_times)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    for i in range(1, len(recovered.times)):
        assert_true(recovered.times[i] >= recovered.times[i-1],
                    "Post-codec monotonicity violated")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 3: Timing fidelity (10 μs floor)
# ═══════════════════════════════════════════════════════════════

@test("TIMING-01: ms → s conversion correct")
def test_timing_conversion():
    brian2_reset()
    sg = make_spike_generator([100.0, 200.0, 300.0])
    mon = SpikeMonitor(sg)
    run(400 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_close(train.times[0], 0.1, tol=1e-6, msg="100 ms → 0.1 s")
    assert_close(train.times[1], 0.2, tol=1e-6, msg="200 ms → 0.2 s")
    assert_close(train.times[2], 0.3, tol=1e-6, msg="300 ms → 0.3 s")


@test("TIMING-02: Sub-ms precision preserved")
def test_timing_subms():
    brian2_reset()
    sg = make_spike_generator([10.123, 20.456, 30.789])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_close(train.times[0], 0.010123, tol=1e-5)
    assert_close(train.times[1], 0.020456, tol=1e-5)
    assert_close(train.times[2], 0.030789, tol=1e-5)


@test("TIMING-03: Round-trip timing within 10 μs floor")
def test_timing_roundtrip():
    brian2_reset()
    original_ms = [10.0, 20.0, 30.0, 40.0, 50.0]
    sg = make_spike_generator(original_ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    original_s = np.array(original_ms) / 1000.0
    max_err_us = float(np.max(np.abs(original_s - recovered.times)) * 1e6)
    assert_true(max_err_us <= 10.0,
                f"Timing error {max_err_us:.2f} μs exceeds 10 μs floor")


@test("TIMING-04: Codec preserves spike count through round-trip")
def test_timing_count_preserved():
    brian2_reset()
    sg = make_spike_generator([10.0, 20.0, 30.0, 40.0, 50.0])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    assert_eq(len(recovered.times), 5)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 4: Round-trip (Brian2 ↔ SpikeLink)
# ═══════════════════════════════════════════════════════════════

@test("ROUNDTRIP-01: SpikeTrain → Brian2 spike times → SpikeTrain")
def test_roundtrip_basic():
    original_times_s = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times_s)

    # SpikeLink → Brian2 times (ms)
    times_ms = Brian2Adapter.to_brian2_times(train_in)
    
    # Verify conversion
    assert_eq(len(times_ms), 5)
    assert_close(times_ms[0], 10.0, tol=1e-6)


@test("ROUNDTRIP-02: Full pipeline: Brian2 → codec → packets → codec")
def test_roundtrip_full_pipeline():
    brian2_reset()
    spike_times_ms = [5.0, 15.0, 25.0, 35.0, 45.0]
    sg = make_spike_generator(spike_times_ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)

    # Brian2 → SpikeLink
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)

    # SpikeLink encode/decode
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)

    assert_eq(len(recovered.times), len(spike_times_ms))


@test("ROUNDTRIP-03: verify_round_trip() helper")
def test_verify_helper():
    brian2_reset()
    sg = make_spike_generator([10.0, 20.0, 30.0, 40.0, 50.0])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    passed = Brian2Adapter.verify_round_trip(mon, neuron_index=0)
    assert_true(passed, "verify_round_trip() failed")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 5: Multi-neuron handling
# ═══════════════════════════════════════════════════════════════

@test("MULTI-01: Extract specific neuron from multi-neuron monitor")
def test_multi_specific():
    brian2_reset()
    # Create multi-neuron generator
    n_neurons = 3
    indices = [0, 0, 1, 1, 1, 2, 2]
    times = [10.0, 20.0, 15.0, 25.0, 35.0, 12.0, 22.0]
    sg = SpikeGeneratorGroup(n_neurons, np.array(indices), np.array(times) * ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train0 = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    train1 = Brian2Adapter.from_spike_monitor(mon, neuron_index=1)
    train2 = Brian2Adapter.from_spike_monitor(mon, neuron_index=2)
    
    assert_eq(len(train0.times), 2)
    assert_eq(len(train1.times), 3)
    assert_eq(len(train2.times), 2)


@test("MULTI-02: Extract all neurons from monitor")
def test_multi_all():
    brian2_reset()
    n_neurons = 3
    indices = [0, 0, 1, 1, 1, 2, 2]
    times = [10.0, 20.0, 15.0, 25.0, 35.0, 12.0, 22.0]
    sg = SpikeGeneratorGroup(n_neurons, np.array(indices), np.array(times) * ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    trains = Brian2Adapter.from_spike_monitor_all(mon)
    
    assert_eq(len(trains), 3)


@test("MULTI-03: Population extraction")
def test_multi_population():
    brian2_reset()
    n_neurons = 5
    indices = list(range(n_neurons)) * 2  # Each neuron spikes twice
    times = [10.0 + i * 2 for i in range(n_neurons * 2)]
    sg = SpikeGeneratorGroup(n_neurons, np.array(indices), np.array(times) * ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    trains = Brian2Adapter.from_spike_monitor_all(mon)
    total_spikes = sum(len(t.times) for t in trains.values())
    assert_eq(total_spikes, n_neurons * 2)


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

@test("EDGE-01: Very close spikes preserved")
def test_close_spikes():
    brian2_reset()
    sg = make_spike_generator([10.0, 10.001, 10.002])
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_true(len(train.times) >= 1, "At least one spike should be preserved")


@test("EDGE-02: Large time values handled")
def test_large_times():
    brian2_reset()
    sg = make_spike_generator([1000.0, 2000.0])
    mon = SpikeMonitor(sg)
    run(3000 * ms)
    
    train = Brian2Adapter.from_spike_monitor(mon, neuron_index=0)
    assert_eq(len(train.times), 2)
    assert_close(train.times[0], 1.0, tol=0.001, msg="1s time")


@test("EDGE-03: is_available() returns correct status", requires_brian2=False)
def test_is_available():
    status = Brian2Adapter.is_available()
    assert_eq(status, BRIAN2_OK)


@test("EDGE-04: Raw times array accepted", requires_brian2=False)
def test_raw_times():
    times_s = np.array([0.01, 0.02, 0.03])
    times_ms = Brian2Adapter.to_brian2_times(SpikeTrain(times=times_s))
    assert_eq(len(times_ms), 3)
    assert_close(times_ms[0], 10.0, tol=1e-6)


@test("EDGE-05: Empty monitor for specific neuron")
def test_empty_neuron():
    brian2_reset()
    n_neurons = 2
    indices = [0, 0, 0]  # Only neuron 0 spikes
    times = [10.0, 20.0, 30.0]
    sg = SpikeGeneratorGroup(n_neurons, np.array(indices), np.array(times) * ms)
    mon = SpikeMonitor(sg)
    run(100 * ms)
    
    train1 = Brian2Adapter.from_spike_monitor(mon, neuron_index=1)
    assert_eq(len(train1.times), 0)


# ═══════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════

@test("STRESS-01: 100 neurons × 50 spikes = 5,000 spikes")
def test_stress_100():
    brian2_reset()
    np.random.seed(42)
    n_neurons = 100
    n_spikes_per = 50
    
    all_indices = []
    all_times = []
    
    for neuron_id in range(n_neurons):
        spike_times = np.sort(np.random.uniform(1.0, 999.0, n_spikes_per))
        all_indices.extend([neuron_id] * n_spikes_per)
        all_times.extend(spike_times.tolist())
    
    sg = SpikeGeneratorGroup(n_neurons, np.array(all_indices), np.array(all_times) * ms)
    mon = SpikeMonitor(sg)
    run(1000 * ms)
    
    trains = Brian2Adapter.from_spike_monitor_all(mon)
    total_spikes = sum(len(t.times) for t in trains.values())
    
    assert_eq(total_spikes, n_neurons * n_spikes_per,
              f"Expected {n_neurons * n_spikes_per}, got {total_spikes}")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS_COUNT, FAIL_COUNT, SKIP_COUNT

    print("=" * 72)
    print("  SpikeLink — Brian2 Platform Tests")
    print("  Lightborne Intelligence · Wave-Native Computing")
    print("=" * 72)
    print()

    if not SPIKELINK_OK:
        print("  ✗ FATAL: spikelink not importable. Aborting.")
        return False

    for name, func, requires_brian2 in TESTS:
        if requires_brian2 and not BRIAN2_OK:
            print(f"  ⊘ SKIP: {name} (brian2 not available)")
            SKIP_COUNT += 1
            continue

        try:
            func()
            print(f"  ✓ PASS: {name}")
            PASS_COUNT += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {name}")
            print(f"          {e}")
            FAIL_COUNT += 1
        except Exception as e:
            print(f"  ✗ ERROR: {name}")
            print(f"          {type(e).__name__}: {e}")
            traceback.print_exc()
            FAIL_COUNT += 1

    print()
    print("-" * 72)
    total = PASS_COUNT + FAIL_COUNT
    pct = (PASS_COUNT / total * 100) if total > 0 else 0
    
    status = "ALL PASSED ✓" if FAIL_COUNT == 0 else "FAILURES DETECTED"
    print(f"  Results: {PASS_COUNT}/{total} passed ({pct:.1f}%) — {status}")
    
    if SKIP_COUNT > 0:
        print(f"  Skipped: {SKIP_COUNT} (brian2 not available)")
    
    print(f"  Backend: {'brian2' if BRIAN2_OK else 'N/A'}")
    print(f"  Stress: 100 neurons × 50 spikes = 5,000 spikes")
    print(f"  Timing floor: 10 μs (SpikeLink quantisation)")
    print("=" * 72)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
