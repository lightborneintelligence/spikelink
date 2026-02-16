#!/usr/bin/env python3
"""
test_platform_pynn.py — SpikeLink ↔ PyNN integration tests
============================================================

Validates the PyNNAdapter against the same invariants enforced
by the existing platform tests (Neo, Brian2, Tonic):

    ✓ Exact spike count preservation
    ✓ Causal ordering (monotonicity)
    ✓ Timing fidelity within SpikeLink 10 μs quantisation floor
    ✓ Round-trip: PyNN → SpikeLink → PyNN
    ✓ Multi-neuron population handling
    ✓ Edge cases & error handling
    ✓ Stress: 380 neurons × 35 spikes (13,300 total)

Requires: pip install spikelink pyNN

Uses PyNN mock backend (no external simulator needed).

Lightborne Intelligence · Dallas TX
"""

import sys
import os
import numpy as np
import traceback

# ── Ensure spikelink is importable ──────────────────────────

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── Imports ──────────────────────────────────────────────────

try:
    from spikelink import SpikeTrain, SpikelinkCodec
    from spikelink.adapters.pynn import PyNNAdapter
    SPIKELINK_OK = True
except ImportError as e:
    print(f"SKIP: spikelink not importable — {e}")
    SPIKELINK_OK = False

try:
    import pyNN.mock as sim
    PYNN_OK = True
except ImportError as e:
    print(f"SKIP: pyNN not importable — {e}")
    PYNN_OK = False


# ── Test Infrastructure ──────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
TESTS = []


def test(name, requires_pynn=True):
    """Register a test function."""
    def decorator(func):
        TESTS.append((name, func, requires_pynn))
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


def pynn_setup():
    sim.setup(timestep=0.1)


def pynn_end():
    try:
        sim.end()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# INVARIANT 1: Exact spike count preservation
# ═══════════════════════════════════════════════════════════════

@test("COUNT-01: Single neuron spike count preserved (5 spikes)")
def test_count_single():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[10.0, 20.0, 30.0, 40.0, 50.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 5, "Spike count mismatch")
    pynn_end()


@test("COUNT-02: Large spike count preserved (1000 spikes)")
def test_count_large():
    pynn_setup()
    spike_times = np.sort(np.random.uniform(0, 10000, 1000)).tolist()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 1000, "Large spike count mismatch")
    pynn_end()


@test("COUNT-03: Empty spike train handled")
def test_count_empty():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 0, "Empty train should yield 0 spikes")
    pynn_end()


@test("COUNT-04: Single spike preserved")
def test_count_one():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[42.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 1)
    pynn_end()


# ═══════════════════════════════════════════════════════════════
# INVARIANT 2: Causal ordering / monotonicity
# ═══════════════════════════════════════════════════════════════

@test("ORDER-01: Output spike times are monotonically increasing")
def test_monotonicity():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[50.0, 10.0, 30.0, 20.0, 40.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    for i in range(1, len(train.times)):
        assert_true(train.times[i] >= train.times[i-1],
                    f"Non-monotonic: {train.times[i-1]} > {train.times[i]}")
    pynn_end()


@test("ORDER-02: Monotonicity preserved through codec round-trip")
def test_monotonicity_codec():
    pynn_setup()
    spike_times = np.random.uniform(0, 1000, 100).tolist()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    for i in range(1, len(recovered.times)):
        assert_true(recovered.times[i] >= recovered.times[i-1],
                    "Post-codec monotonicity violated")
    pynn_end()


# ═══════════════════════════════════════════════════════════════
# INVARIANT 3: Timing fidelity (10 μs floor)
# ═══════════════════════════════════════════════════════════════

@test("TIMING-01: ms → s conversion correct")
def test_timing_conversion():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[100.0, 200.0, 300.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_close(train.times[0], 0.1, tol=1e-6, msg="100 ms → 0.1 s")
    assert_close(train.times[1], 0.2, tol=1e-6, msg="200 ms → 0.2 s")
    assert_close(train.times[2], 0.3, tol=1e-6, msg="300 ms → 0.3 s")
    pynn_end()


@test("TIMING-02: Sub-ms precision preserved")
def test_timing_subms():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[10.123, 20.456, 30.789]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_close(train.times[0], 0.010123, tol=1e-6)
    assert_close(train.times[1], 0.020456, tol=1e-6)
    assert_close(train.times[2], 0.030789, tol=1e-6)
    pynn_end()


@test("TIMING-03: Round-trip timing within 10 μs floor")
def test_timing_roundtrip():
    pynn_setup()
    original_ms = [10.0, 20.0, 30.0, 40.0, 50.0]
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=original_ms))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    original_s = np.array(original_ms) / 1000.0
    max_err_us = float(np.max(np.abs(original_s - recovered.times)) * 1e6)
    assert_true(max_err_us <= 10.0,
                f"Timing error {max_err_us:.2f} μs exceeds 10 μs floor")
    pynn_end()


@test("TIMING-04: Codec preserves spike count through round-trip")
def test_timing_count_preserved():
    pynn_setup()
    spike_times = [10.0, 20.0, 30.0, 40.0, 50.0]
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    assert_eq(len(recovered.times), len(spike_times))
    pynn_end()


# ═══════════════════════════════════════════════════════════════
# INVARIANT 4: Round-trip (PyNN ↔ SpikeLink)
# ═══════════════════════════════════════════════════════════════

@test("ROUNDTRIP-01: SpikeTrain → PyNN → SpikeTrain")
def test_roundtrip_basic():
    pynn_setup()
    original_times_s = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times_s)

    # SpikeLink → PyNN
    pop = PyNNAdapter.to_pynn(train_in, sim_module=sim, label="rt_test")

    # PyNN → SpikeLink
    train_out = PyNNAdapter.from_pynn(pop, neuron_index=0)

    assert_eq(len(train_out.times), len(original_times_s), "Roundtrip count mismatch")

    original = np.sort(np.asarray(original_times_s))
    recovered = np.sort(np.asarray(train_out.times))
    if len(original) > 0:
        max_err_us = float(np.max(np.abs(original - recovered)) * 1e6)
        assert_true(max_err_us <= 10.0,
                    f"Roundtrip timing error {max_err_us:.2f} μs > 10 μs")
    pynn_end()


@test("ROUNDTRIP-02: Full pipeline: PyNN → codec → packets → codec → PyNN")
def test_roundtrip_full_pipeline():
    pynn_setup()
    spike_times_ms = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    pop_in = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_ms))

    # PyNN → SpikeLink SpikeTrain
    train = PyNNAdapter.from_pynn(pop_in, neuron_index=0)

    # SpikeLink encode/decode
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)

    # SpikeLink → PyNN
    pop_out = PyNNAdapter.to_pynn(recovered, sim_module=sim)

    # PyNN → SpikeLink (verify)
    final = PyNNAdapter.from_pynn(pop_out, neuron_index=0)

    assert_eq(len(final.times), len(spike_times_ms))
    pynn_end()


@test("ROUNDTRIP-03: verify_round_trip() helper")
def test_verify_helper():
    pynn_setup()
    passed = PyNNAdapter.verify_round_trip(
        spike_times_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        sim_module=sim,
    )
    assert_true(passed, "verify_round_trip() failed")
    pynn_end()


# ═══════════════════════════════════════════════════════════════
# INVARIANT 5: Multi-neuron handling
# ═══════════════════════════════════════════════════════════════

@test("MULTI-01: Extract specific neuron from population")
def test_multi_specific():
    pynn_setup()
    pop = sim.Population(3, sim.SpikeSourceArray(spike_times=[10.0, 20.0, 30.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=1)
    assert_eq(len(train.times), 3)
    pynn_end()


@test("MULTI-02: from_pynn_block extracts all neurons")
def test_multi_block():
    pynn_setup()
    pop = sim.Population(5, sim.SpikeSourceArray(spike_times=[10.0, 20.0]))
    trains = PyNNAdapter.from_pynn_block(pop)
    assert_eq(len(trains), 5)
    for t in trains:
        assert_eq(len(t.times), 2)
    pynn_end()


@test("MULTI-03: to_pynn_dict creates per-neuron spike dicts")
def test_multi_dict():
    trains = [
        SpikeTrain(times=[0.01, 0.02]),
        SpikeTrain(times=[0.015, 0.025, 0.035]),
    ]
    spike_dict = PyNNAdapter.to_pynn_dict(trains)
    assert_eq(len(spike_dict), 2)
    assert_eq(len(spike_dict[0]), 2)
    assert_eq(len(spike_dict[1]), 3)
    # Values should be in ms
    assert_close(spike_dict[0][0], 10.0, tol=0.01)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 6: Edge cases & error handling
# ═══════════════════════════════════════════════════════════════

@test("EDGE-01: Raw array input accepted", requires_pynn=False)
def test_raw_array():
    train = PyNNAdapter.from_pynn([10.0, 20.0, 30.0])
    assert_eq(len(train.times), 3)
    assert_close(train.times[0], 0.01, tol=1e-6)


@test("EDGE-02: numpy array input accepted", requires_pynn=False)
def test_numpy_array():
    train = PyNNAdapter.from_pynn(np.array([10.0, 20.0, 30.0]))
    assert_eq(len(train.times), 3)


@test("EDGE-03: Very close spikes preserved")
def test_close_spikes():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[10.0, 10.001, 10.002]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 3, "Close spikes must be preserved")
    pynn_end()


@test("EDGE-04: Large time values handled")
def test_large_times():
    pynn_setup()
    pop = sim.Population(1, sim.SpikeSourceArray(spike_times=[100000.0, 200000.0]))
    train = PyNNAdapter.from_pynn(pop, neuron_index=0)
    assert_eq(len(train.times), 2)
    assert_close(train.times[0], 100.0, tol=0.001, msg="100s time")
    pynn_end()


@test("EDGE-05: is_available() returns correct status", requires_pynn=False)
def test_is_available():
    status = PyNNAdapter.is_available()
    assert_eq(status, PYNN_OK)


# ═══════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════

@test("STRESS-01: 380 neurons × 35 spikes = 13,300 spikes")
def test_stress_380():
    pynn_setup()
    np.random.seed(42)
    n_neurons = 380
    n_spikes = 35

    all_trains = []
    for batch_start in range(0, n_neurons, 50):
        batch_size = min(50, n_neurons - batch_start)
        spike_times = np.sort(np.random.uniform(0, 10000, n_spikes)).tolist()
        pop = sim.Population(batch_size, sim.SpikeSourceArray(spike_times=spike_times))
        for i in range(batch_size):
            train = PyNNAdapter.from_pynn(pop, neuron_index=i)
            all_trains.append(train)

    assert_eq(len(all_trains), n_neurons)
    total_spikes = sum(len(t.times) for t in all_trains)
    assert_eq(total_spikes, n_neurons * n_spikes,
              f"Expected {n_neurons * n_spikes}, got {total_spikes}")
    pynn_end()


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS_COUNT, FAIL_COUNT, SKIP_COUNT

    print("=" * 72)
    print("  SpikeLink — PyNN Platform Tests")
    print("  Lightborne Intelligence · Wave-Native Computing")
    print("=" * 72)
    print()

    if not SPIKELINK_OK:
        print("  ✗ FATAL: spikelink not importable. Aborting.")
        return

    for name, func, requires_pynn in TESTS:
        if requires_pynn and not PYNN_OK:
            print(f"  ⊘ SKIP: {name} (pyNN not available)")
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
    total = PASS_COUNT + FAIL_COUNT + SKIP_COUNT
    pct = (PASS_COUNT / (PASS_COUNT + FAIL_COUNT) * 100) if (PASS_COUNT + FAIL_COUNT) > 0 else 0
    
    status = "ALL PASSED ✓" if FAIL_COUNT == 0 else "FAILURES DETECTED"
    print(f"  Results: {PASS_COUNT}/{PASS_COUNT + FAIL_COUNT} passed ({pct:.1f}%) — {status}")
    
    if SKIP_COUNT > 0:
        print(f"  Skipped: {SKIP_COUNT} (pyNN not available)")
    
    print(f"  PyNN backend: {'mock (simulator-independent)' if PYNN_OK else 'N/A'}")
    print(f"  Stress: 380 neurons × 35 spikes = 13,300 spikes")
    print(f"  Timing floor: 10 μs (SpikeLink quantisation)")
    print("=" * 72)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
