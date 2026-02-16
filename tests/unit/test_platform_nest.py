#!/usr/bin/env python3
"""
test_platform_nest.py — SpikeLink ↔ NEST integration tests
============================================================

Validates the NestAdapter against the same invariants enforced
by the existing platform tests (Neo, Brian2, Tonic, PyNN):

    ✓ Exact spike count preservation
    ✓ Causal ordering (monotonicity)
    ✓ Timing fidelity within SpikeLink 10 μs quantisation floor
    ✓ Round-trip: NEST → SpikeLink → NEST
    ✓ Multi-sender spike separation
    ✓ Generator extraction (exact timing, no sim)
    ✓ Edge cases & error handling
    ✓ Stress: 500 generators × 40 spikes (20,000 total)

Auto-detects real NEST vs nest_mock for CI compatibility.

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

# ── Import NEST (real or mock) ───────────────────────────────

try:
    import nest
    NEST_REAL = True
    NEST_OK = True
except ImportError:
    try:
        # Fall back to mock
        from . import nest_mock as nest
    except ImportError:
        try:
            import nest_mock as nest
        except ImportError:
            # Try relative path
            _test_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, _test_dir)
            import nest_mock as nest
    NEST_REAL = False
    NEST_OK = True

# ── Import SpikeLink ─────────────────────────────────────────

try:
    from spikelink import SpikeTrain, SpikelinkCodec
    from spikelink.adapters.nest import NestAdapter
    SPIKELINK_OK = True
except ImportError as e:
    print(f"SKIP: spikelink not importable — {e}")
    SPIKELINK_OK = False
    NEST_OK = False


# ── Test Infrastructure ──────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
TESTS = []


def test(name):
    """Register a test function."""
    def decorator(func):
        TESTS.append((name, func))
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


def nest_reset():
    nest.ResetKernel()


# ═══════════════════════════════════════════════════════════════
# INVARIANT 1: Exact spike count preservation
# ═══════════════════════════════════════════════════════════════

@test("COUNT-01: spike_generator → spike_recorder preserves 5 spikes")
def test_count_basic():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [10.0, 20.0, 30.0, 40.0, 50.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(100.0)
    train = NestAdapter.from_nest(sr)
    assert_eq(len(train.times), 5, "Spike count mismatch")


@test("COUNT-02: Large spike count preserved (500 spikes)")
def test_count_large():
    nest_reset()
    spike_times = np.sort(np.random.uniform(1.0, 9999.0, 500)).tolist()
    sg = nest.Create("spike_generator", params={"spike_times": spike_times})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(10010.0)
    train = NestAdapter.from_nest(sr)
    assert_eq(len(train.times), 500, f"Expected 500, got {len(train.times)}")


@test("COUNT-03: Empty spike_recorder handled")
def test_count_empty():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": []})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(100.0)
    train = NestAdapter.from_nest(sr)
    assert_eq(len(train.times), 0, "Empty recorder should yield 0 spikes")


@test("COUNT-04: Single spike preserved")
def test_count_single():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [42.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(100.0)
    train = NestAdapter.from_nest(sr)
    assert_eq(len(train.times), 1, "Single spike mismatch")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 2: Causal ordering / monotonicity
# ═══════════════════════════════════════════════════════════════

@test("ORDER-01: Output spike times are monotonically increasing")
def test_monotonicity():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [50.0, 10.0, 30.0, 20.0, 40.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(100.0)
    train = NestAdapter.from_nest(sr)
    for i in range(1, len(train.times)):
        assert_true(train.times[i] >= train.times[i-1],
                    f"Non-monotonic: {train.times[i-1]} > {train.times[i]}")


@test("ORDER-02: Monotonicity preserved through codec round-trip")
def test_monotonicity_codec():
    nest_reset()
    spike_times = np.sort(np.random.uniform(1.0, 999.0, 100)).tolist()
    sg = nest.Create("spike_generator", params={"spike_times": spike_times})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(1010.0)
    train = NestAdapter.from_nest(sr)
    
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
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [100.0, 200.0, 300.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(400.0)
    train = NestAdapter.from_nest(sr)
    
    # Note: recorded times include connection delay (default 1ms)
    # So 100ms spike arrives at ~101ms = 0.101s
    assert_true(len(train.times) == 3)
    # Just verify order and rough scale
    assert_true(train.times[0] < train.times[1] < train.times[2])
    assert_true(train.times[0] >= 0.1)  # At least 100ms in seconds


@test("TIMING-02: Generator extraction has exact timing (no delay)")
def test_timing_generator_exact():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [10.0, 20.0, 30.0]})
    train = NestAdapter.from_nest_generator(sg)
    assert_close(train.times[0], 0.01, tol=1e-9, msg="10 ms → 0.01 s exact")
    assert_close(train.times[1], 0.02, tol=1e-9, msg="20 ms → 0.02 s exact")
    assert_close(train.times[2], 0.03, tol=1e-9, msg="30 ms → 0.03 s exact")


@test("TIMING-03: Sub-ms precision preserved in generator")
def test_timing_subms():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [10.123, 20.456, 30.789]})
    train = NestAdapter.from_nest_generator(sg)
    assert_close(train.times[0], 0.010123, tol=1e-9)
    assert_close(train.times[1], 0.020456, tol=1e-9)
    assert_close(train.times[2], 0.030789, tol=1e-9)


@test("TIMING-04: Codec round-trip within 10 μs floor")
def test_timing_codec_roundtrip():
    nest_reset()
    original_ms = [10.0, 20.0, 30.0, 40.0, 50.0]
    sg = nest.Create("spike_generator", params={"spike_times": original_ms})
    train = NestAdapter.from_nest_generator(sg)
    
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    original_s = np.array(original_ms) / 1000.0
    max_err_us = float(np.max(np.abs(original_s - recovered.times)) * 1e6)
    assert_true(max_err_us <= 10.0,
                f"Timing error {max_err_us:.2f} μs exceeds 10 μs floor")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 4: Round-trip (NEST ↔ SpikeLink)
# ═══════════════════════════════════════════════════════════════

@test("ROUNDTRIP-01: SpikeTrain → NEST generator → SpikeTrain (exact)")
def test_roundtrip_generator():
    nest_reset()
    original_times_s = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times_s)

    # SpikeLink → NEST
    sg = NestAdapter.to_nest(train_in, nest_module=nest)

    # NEST → SpikeLink (generator extraction = exact)
    train_out = NestAdapter.from_nest_generator(sg)

    assert_eq(len(train_out.times), len(original_times_s), "Roundtrip count mismatch")
    
    max_err_us = float(np.max(np.abs(np.array(original_times_s) - train_out.times)) * 1e6)
    assert_true(max_err_us <= 0.001,  # Should be exact
                f"Generator roundtrip error {max_err_us:.6f} μs")


@test("ROUNDTRIP-02: SpikeTrain → NEST → simulate → recorder → SpikeTrain")
def test_roundtrip_with_simulation():
    nest_reset()
    original_times_s = [0.01, 0.02, 0.03, 0.04, 0.05]
    train_in = SpikeTrain(times=original_times_s)

    # SpikeLink → NEST
    sg = NestAdapter.to_nest(train_in, nest_module=nest)
    sr = nest.Create("spike_recorder")
    nest.Connect(sg, sr)
    nest.Simulate(100.0)

    # NEST → SpikeLink
    train_out = NestAdapter.from_nest(sr)

    # Count should be preserved (timing includes delay)
    assert_eq(len(train_out.times), len(original_times_s), "Count mismatch")


@test("ROUNDTRIP-03: Full codec pipeline")
def test_roundtrip_full_pipeline():
    nest_reset()
    spike_times_ms = [5.0, 15.0, 25.0, 35.0, 45.0]
    sg = nest.Create("spike_generator", params={"spike_times": spike_times_ms})
    
    # Generator → SpikeTrain
    train = NestAdapter.from_nest_generator(sg)
    
    # SpikeLink encode/decode
    codec = SpikelinkCodec()
    packets = codec.encode_train(train)
    recovered = codec.decode_packets(packets)
    
    # SpikeTrain → NEST → SpikeTrain
    sg2 = NestAdapter.to_nest(recovered, nest_module=nest)
    final = NestAdapter.from_nest_generator(sg2)
    
    assert_eq(len(final.times), len(spike_times_ms))


@test("ROUNDTRIP-04: verify_round_trip() helper")
def test_verify_helper():
    nest_reset()
    passed = NestAdapter.verify_round_trip(
        spike_times_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        nest_module=nest,
    )
    assert_true(passed, "verify_round_trip() failed")


# ═══════════════════════════════════════════════════════════════
# INVARIANT 5: Multi-sender handling
# ═══════════════════════════════════════════════════════════════

@test("MULTI-01: from_nest_events separates by sender GID")
def test_multi_separation():
    nest_reset()
    sg1 = nest.Create("spike_generator", params={"spike_times": [10.0, 20.0]})
    sg2 = nest.Create("spike_generator", params={"spike_times": [15.0, 25.0, 35.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg1, sr)
    nest.Connect(sg2, sr)
    nest.Simulate(100.0)
    
    trains = NestAdapter.from_nest_events(sr)
    
    assert_eq(len(trains), 2, "Should have 2 senders")
    
    # Find counts (GIDs may vary)
    counts = sorted([len(t.times) for t in trains.values()])
    assert_eq(counts, [2, 3], "Spike counts per sender")


@test("MULTI-02: from_nest with sender_gid filter")
def test_multi_filter():
    nest_reset()
    sg1 = nest.Create("spike_generator", params={"spike_times": [10.0, 20.0]})
    sg2 = nest.Create("spike_generator", params={"spike_times": [15.0, 25.0, 35.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg1, sr)
    nest.Connect(sg2, sr)
    nest.Simulate(100.0)
    
    gid1 = sg1.global_id if hasattr(sg1, 'global_id') else sg1._gids[0]
    train1 = NestAdapter.from_nest(sr, sender_gid=gid1)
    
    assert_eq(len(train1.times), 2, "Filtered to sender 1")


@test("MULTI-03: from_nest without filter merges all senders")
def test_multi_merged():
    nest_reset()
    sg1 = nest.Create("spike_generator", params={"spike_times": [10.0, 20.0]})
    sg2 = nest.Create("spike_generator", params={"spike_times": [15.0, 25.0, 35.0]})
    sr = nest.Create("spike_recorder")
    nest.Connect(sg1, sr)
    nest.Connect(sg2, sr)
    nest.Simulate(100.0)
    
    train = NestAdapter.from_nest(sr)  # No sender_gid
    
    assert_eq(len(train.times), 5, "Should have all 5 spikes merged")


# ═══════════════════════════════════════════════════════════════
# GENERATOR EXTRACTION
# ═══════════════════════════════════════════════════════════════

@test("GENERATOR-01: from_nest_generator extracts exact configured times")
def test_generator_extraction():
    nest_reset()
    spike_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    sg = nest.Create("spike_generator", params={"spike_times": spike_times})
    
    train = NestAdapter.from_nest_generator(sg)
    
    assert_eq(len(train.times), 5)
    for i, t in enumerate(spike_times):
        assert_close(train.times[i], t / 1000.0, tol=1e-9)


@test("GENERATOR-02: to_nest_times returns ms")
def test_to_nest_times():
    train = SpikeTrain(times=[0.01, 0.02, 0.03])
    times_ms = NestAdapter.to_nest_times(train)
    
    assert_eq(len(times_ms), 3)
    assert_close(times_ms[0], 10.0, tol=1e-6)
    assert_close(times_ms[1], 20.0, tol=1e-6)
    assert_close(times_ms[2], 30.0, tol=1e-6)


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

@test("EDGE-01: Very close spikes preserved")
def test_close_spikes():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [10.0, 10.001, 10.002]})
    train = NestAdapter.from_nest_generator(sg)
    assert_eq(len(train.times), 3, "Close spikes must be preserved")


@test("EDGE-02: Large time values handled")
def test_large_times():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": [100000.0, 200000.0]})
    train = NestAdapter.from_nest_generator(sg)
    assert_eq(len(train.times), 2)
    assert_close(train.times[0], 100.0, tol=0.001, msg="100s time")


@test("EDGE-03: Empty generator handled")
def test_empty_generator():
    nest_reset()
    sg = nest.Create("spike_generator", params={"spike_times": []})
    train = NestAdapter.from_nest_generator(sg)
    assert_eq(len(train.times), 0)


@test("EDGE-04: is_available() returns correct status")
def test_is_available():
    status = NestAdapter.is_available()
    assert_true(isinstance(status, bool), "is_available should return bool")


@test("EDGE-05: to_nest creates valid generator")
def test_to_nest_creates_generator():
    nest_reset()
    train = SpikeTrain(times=[0.01, 0.02, 0.03])
    sg = NestAdapter.to_nest(train, nest_module=nest)
    
    # Verify it's a NodeCollection with spike_generator
    assert_true(len(sg) == 1)
    params = sg.get()
    assert_eq(params.get("model"), "spike_generator")


# ═══════════════════════════════════════════════════════════════
# STRESS TEST
# ═══════════════════════════════════════════════════════════════

@test("STRESS-01: 500 generators × 40 spikes = 20,000 spikes")
def test_stress_500():
    nest_reset()
    np.random.seed(42)
    n_generators = 500
    n_spikes = 40
    
    all_trains = []
    for _ in range(n_generators):
        spike_times = np.sort(np.random.uniform(1.0, 9999.0, n_spikes)).tolist()
        sg = nest.Create("spike_generator", params={"spike_times": spike_times})
        train = NestAdapter.from_nest_generator(sg)
        all_trains.append(train)
    
    assert_eq(len(all_trains), n_generators)
    total_spikes = sum(len(t.times) for t in all_trains)
    assert_eq(total_spikes, n_generators * n_spikes,
              f"Expected {n_generators * n_spikes}, got {total_spikes}")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def run_all():
    global PASS_COUNT, FAIL_COUNT, SKIP_COUNT

    print("=" * 72)
    print("  SpikeLink — NEST Platform Tests")
    print("  Lightborne Intelligence · Wave-Native Computing")
    print("=" * 72)
    print()

    if not SPIKELINK_OK:
        print("  ✗ FATAL: spikelink not importable. Aborting.")
        return False

    backend = "nest (real)" if NEST_REAL else "nest_mock (faithful mock)"
    print(f"  Backend: {backend}")
    print()

    for name, func in TESTS:
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
    print(f"  Backend: {backend}")
    print(f"  Stress: 500 generators × 40 spikes = 20,000 spikes")
    print(f"  Timing floor: 10 μs (SpikeLink quantisation)")
    print("=" * 72)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
