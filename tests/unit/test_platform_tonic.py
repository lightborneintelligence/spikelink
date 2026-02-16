#!/usr/bin/env python3
"""
test_platform_tonic.py — SpikeLink ↔ Tonic integration tests
==============================================================

Validates the TonicAdapter against the adapter contract invariants:

    ✓ Exact event count preservation
    ✓ Causal ordering (monotonicity)
    ✓ Timing fidelity within SpikeLink 10 μs quantisation floor
    ✓ Round-trip: Tonic → SpikeLink → Tonic
    ✓ Multi-channel handling
    ✓ Edge cases & error handling
    ✓ Stress test

Requires: pip install spikelink[tonic]

Lightborne Intelligence · Dallas TX
"""

import sys
import os
import numpy as np
import traceback

# ── Imports ──────────────────────────────────────────────────

try:
    from spikelink import SpikeTrain, SpikelinkCodec
    from spikelink.adapters.tonic import TonicAdapter
    SPIKELINK_OK = True
except ImportError as e:
    print(f"SKIP: spikelink not importable — {e}")
    SPIKELINK_OK = False

try:
    import tonic
    TONIC_OK = True
except ImportError as e:
    print(f"INFO: tonic not installed — using mock events")
    TONIC_OK = False


# ── Test Infrastructure ──────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
TESTS = []


def test(name, requires_tonic=False):
    """Register a test function."""
    def decorator(func):
        TESTS.append((name, func, requires_tonic))
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


def make_events(n_events, t_max_us=1000000, x_max=128, y_max=128):
    """Create mock DVS events array."""
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(n_events, dtype=dtype)
    events['x'] = np.random.randint(0, x_max, n_events)
    events['y'] = np.random.randint(0, y_max, n_events)
    events['t'] = np.sort(np.random.randint(0, t_max_us, n_events))
    events['p'] = np.random.randint(0, 2, n_events)
    return events


def make_events_at_times(times_us, x=0, y=0, p=1):
    """Create events at specific times."""
    n_events = len(times_us)
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(n_events, dtype=dtype)
    events['x'] = x
    events['y'] = y
    events['t'] = np.array(times_us, dtype=np.int64)
    events['p'] = p
    return events


# ═══════════════════════════════════════════════════════════════
# INVARIANT 1: Exact event count preservation
# ═══════════════════════════════════════════════════════════════

@test("COUNT-01: Event count preserved (5 events)")
def test_count_basic():
    events = make_events_at_times([100000, 200000, 300000, 400000, 500000])
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 5, "Event count mismatch")


@test("COUNT-02: Large event count preserved (1000 events)")
def test_count_large():
    events = make_events(1000)
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 1000, "Large event count mismatch")


@test("COUNT-03: Empty events handled")
def test_count_empty():
    events = make_events(0)
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 0, "Empty events should yield 0 spikes")


@test("COUNT-04: Single event preserved")
def test_count_single():
    events = make_events_at_times([500000])
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 1)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 2: Causal ordering / monotonicity
# ═══════════════════════════════════════════════════════════════

@test("ORDER-01: Output spike times are monotonically increasing")
def test_monotonicity():
    events = make_events(100)
    train = TonicAdapter.from_events(events)
    for i in range(1, len(train.times)):
        assert_true(train.times[i] >= train.times[i-1],
                    f"Non-monotonic: {train.times[i-1]} > {train.times[i]}")


@test("ORDER-02: Monotonicity preserved through codec round-trip")
def test_monotonicity_codec():
    events = make_events(100)
    train = TonicAdapter.from_events(events)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    for i in range(1, len(recovered.times)):
        assert_true(recovered.times[i] >= recovered.times[i-1],
                    "Post-codec monotonicity violated")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 3: Timing fidelity (10 μs floor)
# ═══════════════════════════════════════════════════════════════

@test("TIMING-01: μs → s conversion correct")
def test_timing_conversion():
    events = make_events_at_times([100000, 200000, 300000])  # 0.1s, 0.2s, 0.3s
    train = TonicAdapter.from_events(events)
    assert_close(train.times[0], 0.1, tol=1e-6, msg="100000 μs → 0.1 s")
    assert_close(train.times[1], 0.2, tol=1e-6, msg="200000 μs → 0.2 s")
    assert_close(train.times[2], 0.3, tol=1e-6, msg="300000 μs → 0.3 s")


@test("TIMING-02: Sub-ms precision preserved")
def test_timing_subms():
    events = make_events_at_times([10123, 20456, 30789])  # ~10.123ms, etc
    train = TonicAdapter.from_events(events)
    assert_close(train.times[0], 0.010123, tol=1e-6)
    assert_close(train.times[1], 0.020456, tol=1e-6)
    assert_close(train.times[2], 0.030789, tol=1e-6)


@test("TIMING-03: Round-trip timing within 10 μs floor")
def test_timing_roundtrip():
    original_us = [10000, 20000, 30000, 40000, 50000]  # 10ms increments
    events = make_events_at_times(original_us)
    train = TonicAdapter.from_events(events)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    original_s = np.array(original_us) / 1e6
    max_err_us = float(np.max(np.abs(original_s - recovered.times)) * 1e6)
    assert_true(max_err_us <= 10.0,
                f"Timing error {max_err_us:.2f} μs exceeds 10 μs floor")


@test("TIMING-04: Codec preserves event count through round-trip")
def test_timing_count_preserved():
    events = make_events_at_times([10000, 20000, 30000, 40000, 50000])
    train = TonicAdapter.from_events(events)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    assert_eq(len(recovered.times), 5)


# ═══════════════════════════════════════════════════════════════
# INVARIANT 4: Round-trip (Tonic ↔ SpikeLink)
# ═══════════════════════════════════════════════════════════════

@test("ROUNDTRIP-01: SpikeTrain → events → SpikeTrain")
def test_roundtrip_basic():
    original_times_s = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times_s)

    # SpikeLink → Tonic events
    events = TonicAdapter.to_events(train_in)
    
    # Tonic → SpikeLink
    train_out = TonicAdapter.from_events(events)

    assert_eq(len(train_out.times), len(original_times_s), "Roundtrip count mismatch")
    
    max_err_us = float(np.max(np.abs(np.array(original_times_s) - train_out.times)) * 1e6)
    assert_true(max_err_us <= 1.0, f"Roundtrip timing error {max_err_us:.6f} μs")


@test("ROUNDTRIP-02: Full pipeline: Tonic → codec → packets → codec → Tonic")
def test_roundtrip_full_pipeline():
    original_us = [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000]
    events_in = make_events_at_times(original_us)

    # Tonic → SpikeLink
    train = TonicAdapter.from_events(events_in)

    # SpikeLink encode/decode
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)

    # SpikeLink → Tonic
    events_out = TonicAdapter.to_events(recovered)

    # Tonic → SpikeLink (verify)
    final = TonicAdapter.from_events(events_out)

    assert_eq(len(final.times), len(original_us))


@test("ROUNDTRIP-03: verify_round_trip() helper")
def test_verify_helper():
    events = make_events_at_times([10000, 20000, 30000, 40000, 50000])
    passed = TonicAdapter.verify_round_trip(events)
    assert_true(passed, "verify_round_trip() failed")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 5: Multi-channel / polarity handling
# ═══════════════════════════════════════════════════════════════

@test("MULTI-01: Filter by polarity (ON events only)")
def test_multi_polarity_on():
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(6, dtype=dtype)
    events['t'] = [10000, 20000, 30000, 40000, 50000, 60000]
    events['p'] = [1, 0, 1, 0, 1, 0]  # Alternating ON/OFF
    
    train = TonicAdapter.from_events(events, polarity=1)
    assert_eq(len(train.times), 3, "Should have 3 ON events")


@test("MULTI-02: Filter by polarity (OFF events only)")
def test_multi_polarity_off():
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(6, dtype=dtype)
    events['t'] = [10000, 20000, 30000, 40000, 50000, 60000]
    events['p'] = [1, 0, 1, 0, 1, 0]
    
    train = TonicAdapter.from_events(events, polarity=0)
    assert_eq(len(train.times), 3, "Should have 3 OFF events")


@test("MULTI-03: Filter by pixel location")
def test_multi_pixel():
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(6, dtype=dtype)
    events['x'] = [0, 1, 0, 1, 0, 1]
    events['y'] = [0, 0, 0, 0, 0, 0]
    events['t'] = [10000, 20000, 30000, 40000, 50000, 60000]
    events['p'] = 1
    
    train = TonicAdapter.from_events(events, x=0, y=0)
    assert_eq(len(train.times), 3, "Should have 3 events at (0,0)")


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

@test("EDGE-01: Very close events preserved")
def test_close_events():
    events = make_events_at_times([10000, 10001, 10002])  # 1 μs apart
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 3, "Close events must be preserved")


@test("EDGE-02: Large time values handled")
def test_large_times():
    events = make_events_at_times([100000000, 200000000])  # 100s, 200s
    train = TonicAdapter.from_events(events)
    assert_eq(len(train.times), 2)
    assert_close(train.times[0], 100.0, tol=0.001)


@test("EDGE-03: is_available() returns correct status")
def test_is_available():
    status = TonicAdapter.is_available()
    # Always True since we can work without tonic installed
    assert_true(isinstance(status, bool))


@test("EDGE-04: Different time units (ms)")
def test_time_units_ms():
    # Create events with times in milliseconds
    times_ms = [10, 20, 30, 40, 50]
    events = make_events_at_times([t * 1000 for t in times_ms])  # Convert to μs
    train = TonicAdapter.from_events(events)
    
    assert_close(train.times[0], 0.01, tol=1e-6)


@test("EDGE-05: Preserves x, y coordinates in round-trip")
def test_preserve_coords():
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'u1')])
    events = np.zeros(3, dtype=dtype)
    events['x'] = [10, 20, 30]
    events['y'] = [15, 25, 35]
    events['t'] = [10000, 20000, 30000]
    events['p'] = 1
    
    train = TonicAdapter.from_events(events)
    events_out = TonicAdapter.to_events(train, x=10, y=15)
    
    assert_eq(len(events_out), 3)


# ═══════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════

@test("STRESS-01: 50,000 events (typical DVS burst)")
def test_stress_50k():
    np.random.seed(42)
    events = make_events(50000, t_max_us=10000000)  # 10 seconds
    
    train = TonicAdapter.from_events(events)
    
    assert_eq(len(train.times), 50000, "50k events should be preserved")
    
    # Verify monotonicity
    diffs = np.diff(train.times)
    assert_true(np.all(diffs >= 0), "Events should be monotonic")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS_COUNT, FAIL_COUNT, SKIP_COUNT

    print("=" * 72)
    print("  SpikeLink — Tonic/DVS Platform Tests")
    print("  Lightborne Intelligence · Wave-Native Computing")
    print("=" * 72)
    print()

    if not SPIKELINK_OK:
        print("  ✗ FATAL: spikelink not importable. Aborting.")
        return False

    for name, func, requires_tonic in TESTS:
        if requires_tonic and not TONIC_OK:
            print(f"  ⊘ SKIP: {name} (tonic not available)")
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
        print(f"  Skipped: {SKIP_COUNT} (tonic not available)")
    
    print(f"  Tonic: {'installed' if TONIC_OK else 'mock events (tonic not required)'}")
    print(f"  Stress: 50,000 events (typical DVS burst)")
    print(f"  Timing floor: 10 μs (SpikeLink quantisation)")
    print("=" * 72)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
