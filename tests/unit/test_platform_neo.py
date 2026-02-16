#!/usr/bin/env python3
"""
test_platform_neo.py — SpikeLink ↔ Neo/EBRAINS integration tests
=================================================================

Validates the NeoAdapter against the adapter contract invariants:

    ✓ Exact spike count preservation
    ✓ Causal ordering (monotonicity)
    ✓ Timing fidelity within SpikeLink 10 μs quantisation floor
    ✓ Round-trip: Neo → SpikeLink → Neo
    ✓ Multi-segment handling
    ✓ Edge cases & error handling
    ✓ Stress test

Requires: pip install spikelink[neo]

Lightborne Intelligence · Dallas TX
"""

import sys
import os
import numpy as np
import traceback

# ── Imports ──────────────────────────────────────────────────

try:
    from spikelink import SpikeTrain, SpikelinkCodec
    from spikelink.adapters.neo import NeoAdapter
    SPIKELINK_OK = True
except ImportError as e:
    print(f"SKIP: spikelink not importable — {e}")
    SPIKELINK_OK = False

try:
    import neo
    import quantities as pq
    NEO_OK = True
except ImportError as e:
    print(f"SKIP: neo not importable — {e}")
    NEO_OK = False


# ── Test Infrastructure ──────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
TESTS = []


def test(name, requires_neo=True):
    """Register a test function."""
    def decorator(func):
        TESTS.append((name, func, requires_neo))
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


def make_neo_spiketrain(times_s, t_stop=None):
    """Create a Neo SpikeTrain from times in seconds."""
    times = np.array(times_s) * pq.s
    if t_stop is None:
        t_stop = (max(times_s) + 0.1) * pq.s if len(times_s) > 0 else 1.0 * pq.s
    else:
        t_stop = t_stop * pq.s
    return neo.SpikeTrain(times, t_stop=t_stop)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 1: Exact spike count preservation
# ═══════════════════════════════════════════════════════════════

@test("COUNT-01: Single spike train count preserved (5 spikes)")
def test_count_basic():
    neo_train = make_neo_spiketrain([0.1, 0.2, 0.3, 0.4, 0.5])
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 5, "Spike count mismatch")


@test("COUNT-02: Large spike count preserved (1000 spikes)")
def test_count_large():
    times = np.sort(np.random.uniform(0.0, 10.0, 1000)).tolist()
    neo_train = make_neo_spiketrain(times)
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 1000, "Large spike count mismatch")


@test("COUNT-03: Empty spike train handled")
def test_count_empty():
    neo_train = make_neo_spiketrain([], t_stop=1.0)
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 0, "Empty train should yield 0 spikes")


@test("COUNT-04: Single spike preserved")
def test_count_single():
    neo_train = make_neo_spiketrain([0.5])
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 1)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 2: Causal ordering / monotonicity
# ═══════════════════════════════════════════════════════════════

@test("ORDER-01: Output spike times are monotonically increasing")
def test_monotonicity():
    neo_train = make_neo_spiketrain([0.5, 0.1, 0.3, 0.2, 0.4])
    train = NeoAdapter.from_neo(neo_train)
    for i in range(1, len(train.times)):
        assert_true(train.times[i] >= train.times[i-1],
                    f"Non-monotonic: {train.times[i-1]} > {train.times[i]}")


@test("ORDER-02: Monotonicity preserved through codec round-trip")
def test_monotonicity_codec():
    times = np.sort(np.random.uniform(0.0, 1.0, 100)).tolist()
    neo_train = make_neo_spiketrain(times)
    train = NeoAdapter.from_neo(neo_train)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    for i in range(1, len(recovered.times)):
        assert_true(recovered.times[i] >= recovered.times[i-1],
                    "Post-codec monotonicity violated")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 3: Timing fidelity (10 μs floor)
# ═══════════════════════════════════════════════════════════════

@test("TIMING-01: Neo times preserved in seconds")
def test_timing_conversion():
    neo_train = make_neo_spiketrain([0.1, 0.2, 0.3])
    train = NeoAdapter.from_neo(neo_train)
    assert_close(train.times[0], 0.1, tol=1e-9, msg="0.1s preserved")
    assert_close(train.times[1], 0.2, tol=1e-9, msg="0.2s preserved")
    assert_close(train.times[2], 0.3, tol=1e-9, msg="0.3s preserved")


@test("TIMING-02: Sub-ms precision preserved")
def test_timing_subms():
    neo_train = make_neo_spiketrain([0.010123, 0.020456, 0.030789])
    train = NeoAdapter.from_neo(neo_train)
    assert_close(train.times[0], 0.010123, tol=1e-9)
    assert_close(train.times[1], 0.020456, tol=1e-9)
    assert_close(train.times[2], 0.030789, tol=1e-9)


@test("TIMING-03: Round-trip timing within 10 μs floor")
def test_timing_roundtrip():
    original = [0.01, 0.02, 0.03, 0.04, 0.05]
    neo_train = make_neo_spiketrain(original)
    train = NeoAdapter.from_neo(neo_train)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    max_err_us = float(np.max(np.abs(np.array(original) - recovered.times)) * 1e6)
    assert_true(max_err_us <= 10.0,
                f"Timing error {max_err_us:.2f} μs exceeds 10 μs floor")


@test("TIMING-04: Codec preserves spike count through round-trip")
def test_timing_count_preserved():
    neo_train = make_neo_spiketrain([0.01, 0.02, 0.03, 0.04, 0.05])
    train = NeoAdapter.from_neo(neo_train)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    assert_eq(len(recovered.times), 5)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 4: Round-trip (Neo ↔ SpikeLink)
# ═══════════════════════════════════════════════════════════════

@test("ROUNDTRIP-01: SpikeTrain → Neo → SpikeTrain")
def test_roundtrip_basic():
    original_times = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times)

    # SpikeLink → Neo
    neo_train = NeoAdapter.to_neo(train_in)

    # Neo → SpikeLink
    train_out = NeoAdapter.from_neo(neo_train)

    assert_eq(len(train_out.times), len(original_times), "Roundtrip count mismatch")
    
    max_err_us = float(np.max(np.abs(np.array(original_times) - train_out.times)) * 1e6)
    assert_true(max_err_us <= 0.001, f"Roundtrip timing error {max_err_us:.6f} μs")


@test("ROUNDTRIP-02: Full pipeline: Neo → codec → packets → codec → Neo")
def test_roundtrip_full_pipeline():
    times = [0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075]
    neo_train_in = make_neo_spiketrain(times)

    # Neo → SpikeLink SpikeTrain
    train = NeoAdapter.from_neo(neo_train_in)

    # SpikeLink encode/decode
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)

    # SpikeLink → Neo
    neo_train_out = NeoAdapter.to_neo(recovered)

    # Neo → SpikeLink (verify)
    final = NeoAdapter.from_neo(neo_train_out)

    assert_eq(len(final.times), len(times))


@test("ROUNDTRIP-03: verify_round_trip() helper")
def test_verify_helper():
    neo_train = make_neo_spiketrain([0.01, 0.02, 0.03, 0.04, 0.05])
    passed = NeoAdapter.verify_round_trip(neo_train)
    assert_true(passed, "verify_round_trip() failed")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 5: Multi-segment / block handling
# ═══════════════════════════════════════════════════════════════

@test("MULTI-01: Extract from Neo Block with multiple segments")
def test_multi_block():
    block = neo.Block()
    seg1 = neo.Segment()
    seg1.spiketrains.append(make_neo_spiketrain([0.1, 0.2, 0.3]))
    seg2 = neo.Segment()
    seg2.spiketrains.append(make_neo_spiketrain([0.4, 0.5]))
    block.segments.append(seg1)
    block.segments.append(seg2)
    
    trains = NeoAdapter.from_neo_block(block)
    assert_eq(len(trains), 2, "Should extract from 2 segments")


@test("MULTI-02: Extract specific segment")
def test_multi_segment():
    block = neo.Block()
    seg1 = neo.Segment()
    seg1.spiketrains.append(make_neo_spiketrain([0.1, 0.2]))
    seg2 = neo.Segment()
    seg2.spiketrains.append(make_neo_spiketrain([0.3, 0.4, 0.5]))
    block.segments.append(seg1)
    block.segments.append(seg2)
    
    train = NeoAdapter.from_neo(block.segments[1].spiketrains[0])
    assert_eq(len(train.times), 3)


@test("MULTI-03: Multiple spike trains in segment")
def test_multi_trains():
    seg = neo.Segment()
    seg.spiketrains.append(make_neo_spiketrain([0.1, 0.2]))
    seg.spiketrains.append(make_neo_spiketrain([0.15, 0.25, 0.35]))
    
    train1 = NeoAdapter.from_neo(seg.spiketrains[0])
    train2 = NeoAdapter.from_neo(seg.spiketrains[1])
    
    assert_eq(len(train1.times), 2)
    assert_eq(len(train2.times), 3)


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

@test("EDGE-01: Very close spikes preserved")
def test_close_spikes():
    neo_train = make_neo_spiketrain([0.01, 0.010001, 0.010002])
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 3, "Close spikes must be preserved")


@test("EDGE-02: Large time values handled")
def test_large_times():
    neo_train = make_neo_spiketrain([100.0, 200.0])
    train = NeoAdapter.from_neo(neo_train)
    assert_eq(len(train.times), 2)
    assert_close(train.times[0], 100.0, tol=0.001)


@test("EDGE-03: Units conversion (ms input)")
def test_units_ms():
    # Neo with millisecond units
    times = np.array([10.0, 20.0, 30.0]) * pq.ms
    neo_train = neo.SpikeTrain(times, t_stop=100.0 * pq.ms)
    train = NeoAdapter.from_neo(neo_train)
    # Should be converted to seconds
    assert_close(train.times[0], 0.01, tol=1e-6)
    assert_close(train.times[1], 0.02, tol=1e-6)


@test("EDGE-04: is_available() returns correct status", requires_neo=False)
def test_is_available():
    status = NeoAdapter.is_available()
    assert_eq(status, NEO_OK)


@test("EDGE-05: t_start/t_stop preserved")
def test_time_bounds():
    times = np.array([0.1, 0.2, 0.3]) * pq.s
    neo_train = neo.SpikeTrain(times, t_start=0.0 * pq.s, t_stop=1.0 * pq.s)
    train = NeoAdapter.from_neo(neo_train)
    assert_close(train.t_start, 0.0, tol=1e-9)
    assert_close(train.t_stop, 1.0, tol=1e-9)


# ═══════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════

@test("STRESS-01: 500 spike trains × 50 spikes = 25,000 spikes")
def test_stress_500():
    np.random.seed(42)
    n_trains = 500
    n_spikes = 50
    
    all_trains = []
    for _ in range(n_trains):
        times = np.sort(np.random.uniform(0.0, 10.0, n_spikes)).tolist()
        neo_train = make_neo_spiketrain(times)
        train = NeoAdapter.from_neo(neo_train)
        all_trains.append(train)
    
    assert_eq(len(all_trains), n_trains)
    total_spikes = sum(len(t.times) for t in all_trains)
    assert_eq(total_spikes, n_trains * n_spikes,
              f"Expected {n_trains * n_spikes}, got {total_spikes}")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS_COUNT, FAIL_COUNT, SKIP_COUNT

    print("=" * 72)
    print("  SpikeLink — Neo/EBRAINS Platform Tests")
    print("  Lightborne Intelligence · Wave-Native Computing")
    print("=" * 72)
    print()

    if not SPIKELINK_OK:
        print("  ✗ FATAL: spikelink not importable. Aborting.")
        return False

    for name, func, requires_neo in TESTS:
        if requires_neo and not NEO_OK:
            print(f"  ⊘ SKIP: {name} (neo not available)")
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
        print(f"  Skipped: {SKIP_COUNT} (neo not available)")
    
    print(f"  Backend: {'neo + quantities' if NEO_OK else 'N/A'}")
    print(f"  Stress: 500 trains × 50 spikes = 25,000 spikes")
    print(f"  Timing floor: 10 μs (SpikeLink quantisation)")
    print("=" * 72)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
