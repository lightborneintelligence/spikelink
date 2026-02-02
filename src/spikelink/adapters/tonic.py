#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  PLATFORM TEST #3: Tonic (Event-Driven Neuromorphic Datasets)
═══════════════════════════════════════════════════════════════════════════
  Lightborne Intelligence

  Proves SpikeLink v2 can faithfully transport real event-camera data
  as exposed by Tonic (PyTorch Vision/Audio for neuromorphic data).

  Two modes, both reported:

    Mode A — Timing-Exact (official platform gate):
      spikes_per_packet=1, each event carries its exact timestamp
      via the absolute chunk_start_10us anchor.
      Gate: count exact + time ≤10μs + ordering exact.

    Mode B — Throughput Trade (v2 meaning story):
      spikes_per_packet=7 (default), larger chunks.
      Reports intra-chunk timing distortion as a function of
      chunk size. Shows bounded tradeoff, not hidden error.

  Event → spike mapping (AER-style):
    neuron_id = (y * W + x) * 2 + p
    spike_time = t (in seconds)
    amplitude = 1.0

  Timing precision note:
    chunk_start_10us encodes absolute time at 10μs resolution.
    Mode A validation budget: |Δt| ≤ 10μs per event.

  Test datasets (synthetic, matching real formats exactly):
    - NMNIST-format:    34×34 pixels, 2 polarities, ~5000 events
    - DVSGesture-format: 128×128 pixels, 2 polarities, ~50000 events

  Dependencies: tonic (for dtype/transform verification), numpy
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sys
import os
import time as time_mod

sys.path.insert(0, os.path.dirname(__file__))
from spikelink_v2 import SpikelinkCodecV2, SpikeTrain as V2SpikeTrain

# Tonic — for dtype verification and transforms
import tonic


# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════

# Tonic canonical event dtype
TONIC_DTYPE = np.dtype([('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')])

# NMNIST sensor: 34×34 pixels
NMNIST_W, NMNIST_H = 34, 34

# DVSGesture sensor: 128×128 pixels
DVS_W, DVS_H = 128, 128


# ═════════════════════════════════════════════════════════════════════════
# AER EVENT ↔ V2 ADAPTER
# ═════════════════════════════════════════════════════════════════════════

class TonicAdapterV2:
    """
    Bridge between Tonic event arrays and SpikeLink v2.

    Tonic events: structured array with (x, y, t, p)
      - t in microseconds (integer)
      - x, y pixel coordinates
      - p polarity (0 or 1)

    SpikeLink v2: SpikeTrain with (times, amplitudes)
      - times in seconds (float64)
      - amplitudes (float64)

    AER mapping: neuron_id = (y * W + x) * 2 + p
    This is deterministic and reversible.
    """

    def __init__(self, sensor_width: int, sensor_height: int):
        self.W = sensor_width
        self.H = sensor_height
        self.n_channels = sensor_width * sensor_height * 2  # ×2 for polarity

    def events_to_spike_trains(self, events: np.ndarray) -> dict:
        """Convert Tonic events → per-channel V2SpikeTrains.

        Returns dict mapping neuron_id → V2SpikeTrain.
        Only channels with ≥1 event are included.
        """
        # AER address encoding
        neuron_ids = (events['y'] * self.W + events['x']) * 2 + events['p']
        times_s = events['t'].astype(np.float64) * 1e-6  # μs → seconds

        trains = {}
        unique_ids = np.unique(neuron_ids)
        for nid in unique_ids:
            mask = neuron_ids == nid
            ch_times = np.sort(times_s[mask])
            trains[int(nid)] = V2SpikeTrain(
                times=ch_times, amplitudes=np.ones(len(ch_times)))
        return trains

    def spike_trains_to_events(self, trains: dict,
                               original_events: np.ndarray) -> np.ndarray:
        """Convert per-channel V2SpikeTrains → Tonic events.

        Uses original_events only for channel→(x,y,p) reverse mapping.
        Times come from the recovered spike trains.
        """
        all_events = []
        for nid, train in trains.items():
            # Reverse AER: nid = (y*W + x)*2 + p
            p = nid % 2
            spatial = nid // 2
            x = spatial % self.W
            y = spatial // self.W

            for t_s in train.times:
                t_us = int(round(t_s * 1e6))
                all_events.append((x, y, t_us, p))

        if not all_events:
            return np.zeros(0, dtype=TONIC_DTYPE)

        result = np.array(all_events, dtype=TONIC_DTYPE)
        # Sort by time (same as original ordering)
        result = np.sort(result, order='t')
        return result


# ═════════════════════════════════════════════════════════════════════════
# SYNTHETIC EVENT GENERATORS (matching real dataset formats)
# ═════════════════════════════════════════════════════════════════════════

def generate_nmnist_events(digit: int = 3, seed: int = 42) -> np.ndarray:
    """Generate NMNIST-format events: 34×34, saccade pattern.

    Real NMNIST: MNIST digits shown to a DVS camera undergoing
    3 micro-saccades. Each saccade generates a burst of events
    on digit edges. ~4000-8000 events per sample, ~300ms duration.
    """
    rng = np.random.RandomState(seed + digit)

    # Create a simple digit mask (edges of a rough shape)
    mask = np.zeros((NMNIST_H, NMNIST_W), dtype=bool)
    # Draw digit-like strokes
    cx, cy = 17, 17
    for angle in np.linspace(0, 2 * np.pi * (digit + 1) / 10, 20 + digit * 3):
        r = 8 + rng.randint(-2, 3)
        px = int(cx + r * np.cos(angle))
        py = int(cy + r * np.sin(angle))
        if 0 <= px < NMNIST_W and 0 <= py < NMNIST_H:
            mask[py, px] = True
            # Thicken
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < NMNIST_W and 0 <= ny < NMNIST_H:
                    mask[ny, nx] = True

    active_pixels = np.argwhere(mask)  # (y, x) pairs
    n_active = len(active_pixels)

    # 3 saccades, each ~100ms, with inter-saccade gaps
    events = []
    t_us = 0
    for saccade in range(3):
        # Saccade onset
        saccade_start = t_us
        n_events_saccade = rng.randint(1200, 2200)

        for _ in range(n_events_saccade):
            # Pick an active pixel
            idx = rng.randint(0, n_active)
            y, x = active_pixels[idx]
            # Time within saccade (concentrated at edges → burst-like)
            dt = int(rng.exponential(50))  # μs
            t_us = saccade_start + dt + rng.randint(0, 100000)
            p = rng.randint(0, 2)
            events.append((x, y, t_us, p))

        # Inter-saccade gap
        t_us += 100000  # 100ms

    events = np.array(events, dtype=TONIC_DTYPE)
    events = np.sort(events, order='t')
    # Normalize timestamps to start at 0
    events['t'] -= events['t'].min()
    return events


def generate_dvsgesture_events(gesture_id: int = 1, seed: int = 42) -> np.ndarray:
    """Generate DVSGesture-format events: 128×128, hand gesture motion.

    Real DVSGesture: 11 gesture classes, 128×128 sensor, ~1-6 seconds,
    10k-100k events. Hand motion creates spatiotemporal event streams.
    """
    rng = np.random.RandomState(seed + gesture_id)

    # Simulate hand motion trajectory
    duration_us = 1500000  # 1.5s
    n_events = 50000

    # Hand center trajectory (smooth motion)
    n_keyframes = 10
    kf_x = rng.randint(30, 98, n_keyframes).astype(float)
    kf_y = rng.randint(30, 98, n_keyframes).astype(float)
    kf_t = np.linspace(0, duration_us, n_keyframes)

    events = []
    for i in range(n_events):
        # Random time
        t = rng.randint(0, duration_us)
        # Interpolate hand center at this time
        cx = np.interp(t, kf_t, kf_x)
        cy = np.interp(t, kf_t, kf_y)

        # Event near hand with spatial spread
        spread = 15 + rng.exponential(5)
        x = int(cx + rng.randn() * spread) % DVS_W
        y = int(cy + rng.randn() * spread) % DVS_H
        p = rng.randint(0, 2)
        events.append((x, y, t, p))

    events = np.array(events, dtype=TONIC_DTYPE)
    events = np.sort(events, order='t')
    events['t'] -= events['t'].min()
    return events


# ═════════════════════════════════════════════════════════════════════════
# TEST HARNESS
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

    def section(self, title: str):
        self.tests.append((f"── {title} ──", "SECTION", ""))

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append(f"  PLATFORM TEST: {self.name}")
        lines.append(f"  Lightborne Intelligence — SpikeLink v2.0")
        lines.append("=" * 72)
        for name, status, detail in self.tests:
            if status == "SECTION":
                lines.append(f"\n  {name}")
                continue
            icon = "✅" if status == "PASS" else "❌"
            line = f"  {icon} {name}"
            if detail:
                line += f"  │  {detail}"
            lines.append(line)
        lines.append("-" * 72)
        total = self.passed + self.failed
        lines.append(f"  Results: {self.passed}/{total} passed")
        if self.failed == 0:
            lines.append(f"  ✅ {self.name} — FULLY COMPATIBLE")
        else:
            lines.append(f"  ❌ {self.name} — {self.failed} FAILURES")
        lines.append("=" * 72)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# TRANSPORT PIPELINE (shared by Mode A and Mode B)
# ═════════════════════════════════════════════════════════════════════════

def transport_events(events: np.ndarray, adapter: TonicAdapterV2,
                     spikes_per_packet: int = 1) -> tuple:
    """Full pipeline: events → per-channel v2 trains → encode → decode → events.

    Returns: (recovered_events, metrics_dict)
    """
    # Events → per-channel spike trains
    trains_in = adapter.events_to_spike_trains(events)

    # Encode/decode each channel
    codec = SpikelinkCodecV2(
        spikes_per_packet=spikes_per_packet,
        max_amplitude=2.0)

    trains_out = {}
    total_packets = 0

    for nid, train_in in trains_in.items():
        if len(train_in.times) == 0:
            trains_out[nid] = train_in
            continue
        codec.reset()
        pkts = codec.encode_train(train_in)
        total_packets += len(pkts)
        trains_out[nid] = codec.decode_packets(pkts)

    # Reconstruct events
    recovered = adapter.spike_trains_to_events(trains_out, events)

    metrics = {
        'total_packets': total_packets,
        'n_channels_in': len(trains_in),
        'n_channels_out': len(trains_out),
    }

    return recovered, metrics


# ═════════════════════════════════════════════════════════════════════════
# INVARIANT VALIDATORS
# ═════════════════════════════════════════════════════════════════════════

def check_hard_invariants(results: PlatformTestResult,
                          events_in: np.ndarray,
                          events_out: np.ndarray,
                          prefix: str):
    """Hard invariants: count, ordering, range conservation."""

    # Event count
    results.check(
        f"{prefix}: event count exact",
        len(events_out) == len(events_in),
        f"In: {len(events_in)}, Out: {len(events_out)}"
    )

    # Temporal ordering (monotonic)
    if len(events_out) > 1:
        t_diffs = np.diff(events_out['t'])
        results.check(
            f"{prefix}: time monotonic (causal ordering)",
            np.all(t_diffs >= 0),
            f"Min Δt: {t_diffs.min()} μs"
        )

    # Spatial range conservation
    if len(events_out) > 0 and len(events_in) > 0:
        x_match = (events_out['x'].min() >= 0 and
                   events_out['x'].max() <= events_in['x'].max())
        y_match = (events_out['y'].min() >= 0 and
                   events_out['y'].max() <= events_in['y'].max())
        results.check(
            f"{prefix}: spatial range conserved",
            x_match and y_match,
            f"x:[{events_out['x'].min()},{events_out['x'].max()}], "
            f"y:[{events_out['y'].min()},{events_out['y'].max()}]"
        )

    # Polarity set conserved
    if len(events_out) > 0:
        p_in = set(np.unique(events_in['p']))
        p_out = set(np.unique(events_out['p']))
        results.check(
            f"{prefix}: polarity set conserved",
            p_out == p_in,
            f"In: {p_in}, Out: {p_out}"
        )


def check_timing_invariants(results: PlatformTestResult,
                            events_in: np.ndarray,
                            events_out: np.ndarray,
                            prefix: str,
                            mode: str,
                            timing_budget_us: float = 10.0):
    """Timing invariants: Mode A (exact) or Mode B (bounded)."""

    if len(events_in) != len(events_out) or len(events_in) == 0:
        results.check(f"{prefix}: timing validation",
                      False, "Count mismatch — cannot compare timing")
        return

    t_in = events_in['t'].astype(np.float64)
    t_out = events_out['t'].astype(np.float64)

    if mode == "A":
        # Mode A: each event should be within 10μs of original
        # chunk_start_10us has 10μs resolution → max quantization error = 10μs
        abs_errors = np.abs(t_in - t_out)
        max_err = abs_errors.max()
        mean_err = abs_errors.mean()
        p99_err = np.percentile(abs_errors, 99)

        results.check(
            f"{prefix}: max |Δt| ≤ {timing_budget_us:.0f}μs (10μs quantization)",
            max_err <= timing_budget_us,
            f"Max: {max_err:.1f}μs, Mean: {mean_err:.1f}μs, P99: {p99_err:.1f}μs"
        )
    else:
        # Mode B: report timing distortion, don't gate on it
        abs_errors = np.abs(t_in - t_out)
        max_err = abs_errors.max()
        mean_err = abs_errors.mean()
        p99_err = np.percentile(abs_errors, 99)

        results.check(
            f"{prefix}: timing distortion reported (not gated)",
            True,
            f"Max: {max_err:.0f}μs, Mean: {mean_err:.0f}μs, P99: {p99_err:.0f}μs"
        )


def check_distribution_invariants(results: PlatformTestResult,
                                  events_in: np.ndarray,
                                  events_out: np.ndarray,
                                  adapter: TonicAdapterV2,
                                  prefix: str):
    """Distribution invariants: per-channel counts, IEI, polarity balance."""

    if len(events_in) == 0 or len(events_out) == 0:
        return

    # Per-channel spike count correlation
    nids_in = (events_in['y'] * adapter.W + events_in['x']) * 2 + events_in['p']
    nids_out = (events_out['y'] * adapter.W + events_out['x']) * 2 + events_out['p']

    unique_in, counts_in = np.unique(nids_in, return_counts=True)
    unique_out, counts_out = np.unique(nids_out, return_counts=True)

    # Build count vectors on shared channels
    all_nids = np.union1d(unique_in, unique_out)
    vec_in = np.zeros(len(all_nids))
    vec_out = np.zeros(len(all_nids))

    in_map = dict(zip(unique_in, counts_in))
    out_map = dict(zip(unique_out, counts_out))

    for i, nid in enumerate(all_nids):
        vec_in[i] = in_map.get(nid, 0)
        vec_out[i] = out_map.get(nid, 0)

    if len(all_nids) > 2:
        count_corr = np.corrcoef(vec_in, vec_out)[0, 1]
        results.check(
            f"{prefix}: per-channel count correlation > 0.99",
            count_corr > 0.99,
            f"r = {count_corr:.6f} across {len(all_nids)} channels"
        )

    # Channel count exact match
    results.check(
        f"{prefix}: active channel count exact",
        len(unique_in) == len(unique_out),
        f"In: {len(unique_in)}, Out: {len(unique_out)}"
    )

    # Polarity balance
    on_in = np.sum(events_in['p'] == 1)
    on_out = np.sum(events_out['p'] == 1)
    off_in = np.sum(events_in['p'] == 0)
    off_out = np.sum(events_out['p'] == 0)

    results.check(
        f"{prefix}: polarity balance exact",
        on_in == on_out and off_in == off_out,
        f"ON: {on_in}→{on_out}, OFF: {off_in}→{off_out}"
    )

    # Inter-event interval distribution (global)
    iei_in = np.diff(events_in['t'].astype(np.float64))
    iei_out = np.diff(events_out['t'].astype(np.float64))

    if len(iei_in) > 10 and len(iei_out) > 10:
        mean_iei_err = abs(np.mean(iei_in) - np.mean(iei_out)) / (np.mean(iei_in) + 1e-12)
        results.check(
            f"{prefix}: mean IEI preserved (<5%)",
            mean_iei_err < 0.05,
            f"In: {np.mean(iei_in):.1f}μs, Out: {np.mean(iei_out):.1f}μs, "
            f"Err: {mean_iei_err:.2%}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 1: TONIC FORMAT COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════

def test_tonic_format(results: PlatformTestResult):
    print("\n[1] Tonic Format Compatibility...")

    # Verify our synthetic data matches Tonic's canonical dtype
    events = generate_nmnist_events(digit=3)

    results.check(
        "Synthetic events match Tonic dtype",
        events.dtype == TONIC_DTYPE,
        f"dtype: {events.dtype}"
    )

    # Verify Tonic transforms work on our data
    from tonic.transforms import TimeAlignment, Denoise, Downsample

    aligned = TimeAlignment()(events.copy())
    results.check(
        "Tonic TimeAlignment works on our events",
        len(aligned) > 0 and aligned['t'].min() == 0,
        f"{len(aligned)} events, t_min={aligned['t'].min()}"
    )

    # Verify AER mapping is reversible
    adapter = TonicAdapterV2(NMNIST_W, NMNIST_H)
    trains = adapter.events_to_spike_trains(events)
    recovered = adapter.spike_trains_to_events(trains, events)

    results.check(
        "AER mapping round-trip preserves event count",
        len(recovered) == len(events),
        f"In: {len(events)}, Out: {len(recovered)}"
    )


# ═════════════════════════════════════════════════════════════════════════
# TEST 2: MODE A — NMNIST (timing-exact, official gate)
# ═════════════════════════════════════════════════════════════════════════

def test_mode_a_nmnist(results: PlatformTestResult):
    print("[2] Mode A — NMNIST (timing-exact, spikes_per_packet=1)...")

    events = generate_nmnist_events(digit=5)
    adapter = TonicAdapterV2(NMNIST_W, NMNIST_H)

    recovered, metrics = transport_events(events, adapter, spikes_per_packet=1)

    results.section("NMNIST Mode A — Hard Invariants")
    check_hard_invariants(results, events, recovered, "NMNIST-A")

    results.section("NMNIST Mode A — Timing (10μs budget)")
    check_timing_invariants(results, events, recovered, "NMNIST-A", mode="A")

    results.section("NMNIST Mode A — Distribution")
    check_distribution_invariants(results, events, recovered, adapter, "NMNIST-A")

    results.check(
        "NMNIST-A: transport efficiency",
        True,
        f"{len(events)} events → {metrics['total_packets']} packets "
        f"({metrics['n_channels_in']} channels)"
    )


# ═════════════════════════════════════════════════════════════════════════
# TEST 3: MODE A — DVSGesture (timing-exact at scale)
# ═════════════════════════════════════════════════════════════════════════

def test_mode_a_dvsgesture(results: PlatformTestResult):
    print("[3] Mode A — DVSGesture (timing-exact at scale)...")

    events = generate_dvsgesture_events(gesture_id=3)
    adapter = TonicAdapterV2(DVS_W, DVS_H)

    t_start = time_mod.time()
    recovered, metrics = transport_events(events, adapter, spikes_per_packet=1)
    elapsed = time_mod.time() - t_start

    results.section("DVSGesture Mode A — Hard Invariants")
    check_hard_invariants(results, events, recovered, "DVSGesture-A")

    results.section("DVSGesture Mode A — Timing (10μs budget)")
    check_timing_invariants(results, events, recovered, "DVSGesture-A", mode="A")

    results.section("DVSGesture Mode A — Distribution")
    check_distribution_invariants(results, events, recovered, adapter, "DVSGesture-A")

    results.check(
        "DVSGesture-A: scale + speed",
        True,
        f"{len(events)} events in {elapsed:.2f}s "
        f"({len(events)/elapsed:.0f} events/s), "
        f"{metrics['total_packets']} packets"
    )


# ═════════════════════════════════════════════════════════════════════════
# TEST 4: MODE B — NMNIST (chunked, trade-off report)
# ═════════════════════════════════════════════════════════════════════════

def test_mode_b_nmnist(results: PlatformTestResult):
    print("[4] Mode B — NMNIST (chunked, timing trade-off)...")

    events = generate_nmnist_events(digit=7)
    adapter = TonicAdapterV2(NMNIST_W, NMNIST_H)

    results.section("NMNIST Mode B — Chunk Size Sweep")

    for spp in [3, 7, 16]:
        recovered, metrics = transport_events(events, adapter,
                                              spikes_per_packet=spp)
        # Hard invariants still required
        count_ok = len(recovered) == len(events)

        # Timing distortion measurement
        if count_ok and len(events) > 0:
            t_in = events['t'].astype(np.float64)
            t_out = recovered['t'].astype(np.float64)
            abs_errors = np.abs(t_in - t_out)
            max_err = abs_errors.max()
            mean_err = abs_errors.mean()
            p99_err = np.percentile(abs_errors, 99)
        else:
            max_err = mean_err = p99_err = float('inf')

        results.check(
            f"NMNIST-B (spp={spp}): count exact + timing reported",
            count_ok,
            f"Count: {len(recovered)}/{len(events)}, "
            f"Δt max={max_err:.0f}μs mean={mean_err:.0f}μs p99={p99_err:.0f}μs, "
            f"pkts={metrics['total_packets']}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 5: MODE B — DVSGesture (chunked at scale, trade-off report)
# ═════════════════════════════════════════════════════════════════════════

def test_mode_b_dvsgesture(results: PlatformTestResult):
    print("[5] Mode B — DVSGesture (chunked at scale)...")

    events = generate_dvsgesture_events(gesture_id=5)
    adapter = TonicAdapterV2(DVS_W, DVS_H)

    results.section("DVSGesture Mode B — Chunk Size Sweep")

    for spp in [3, 7, 16]:
        t_start = time_mod.time()
        recovered, metrics = transport_events(events, adapter,
                                              spikes_per_packet=spp)
        elapsed = time_mod.time() - t_start

        count_ok = len(recovered) == len(events)

        if count_ok and len(events) > 0:
            t_in = events['t'].astype(np.float64)
            t_out = recovered['t'].astype(np.float64)
            abs_errors = np.abs(t_in - t_out)
            max_err = abs_errors.max()
            mean_err = abs_errors.mean()
        else:
            max_err = mean_err = float('inf')

        results.check(
            f"DVSGesture-B (spp={spp}): count + timing + speed",
            count_ok,
            f"Count: {len(recovered)}/{len(events)}, "
            f"Δt max={max_err:.0f}μs mean={mean_err:.0f}μs, "
            f"{elapsed:.2f}s ({len(events)/elapsed:.0f} ev/s)"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 6: MULTI-SAMPLE BATCH (5 NMNIST digits)
# ═════════════════════════════════════════════════════════════════════════

def test_multi_sample(results: PlatformTestResult):
    print("[6] Multi-Sample Batch (5 NMNIST digits, Mode A)...")

    adapter = TonicAdapterV2(NMNIST_W, NMNIST_H)
    total_in = 0
    total_out = 0
    all_timing_errors = []

    for digit in range(5):
        events = generate_nmnist_events(digit=digit)
        recovered, _ = transport_events(events, adapter, spikes_per_packet=1)
        total_in += len(events)
        total_out += len(recovered)

        if len(events) == len(recovered) and len(events) > 0:
            t_err = np.abs(
                events['t'].astype(np.float64) -
                recovered['t'].astype(np.float64))
            all_timing_errors.append(t_err.max())

    results.check(
        "5-digit batch: total event count exact",
        total_out == total_in,
        f"Total: {total_in} → {total_out}"
    )

    if all_timing_errors:
        worst_timing = max(all_timing_errors)
        results.check(
            "5-digit batch: worst-case timing ≤ 10μs",
            worst_timing <= 10.0,
            f"Worst max |Δt| across 5 samples: {worst_timing:.1f}μs"
        )


# ═════════════════════════════════════════════════════════════════════════
# TEST 7: TONIC TRANSFORM CHAIN COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════

def test_transform_chain(results: PlatformTestResult):
    print("[7] Tonic Transform Chain Compatibility...")

    events_raw = generate_dvsgesture_events(gesture_id=2)
    adapter = TonicAdapterV2(DVS_W, DVS_H)

    # Apply standard Tonic preprocessing
    from tonic.transforms import Compose, TimeAlignment, Denoise

    transform = Compose([
        TimeAlignment(),
        Denoise(filter_time=10000),  # 10ms refractory filter
    ])

    events_clean = transform(events_raw.copy())
    n_before = len(events_raw)
    n_after = len(events_clean)

    # Transport the cleaned events through v2 (Mode A)
    recovered, metrics = transport_events(events_clean, adapter,
                                          spikes_per_packet=1)

    results.check(
        "Tonic Denoise → SpikeLink v2: count exact post-transform",
        len(recovered) == len(events_clean),
        f"Raw: {n_before} → Denoised: {n_after} → Transported: {len(recovered)}"
    )

    if len(recovered) == len(events_clean) and len(events_clean) > 0:
        t_err = np.abs(
            events_clean['t'].astype(np.float64) -
            recovered['t'].astype(np.float64))
        results.check(
            "Tonic Denoise → SpikeLink v2: timing ≤ 10μs",
            t_err.max() <= 10.0,
            f"Max |Δt|: {t_err.max():.1f}μs"
        )


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    results = PlatformTestResult(
        "Tonic (Event-Driven Neuromorphic Datasets)")

    print("\n" + "=" * 72)
    print("  SPIKELINK v2.0 — PLATFORM INTEROP TEST #3")
    print("  Target: Tonic Event-Camera Data")
    print(f"  tonic {tonic.__version__}")
    print()
    print("  Timing precision: chunk_start_10us → 10μs resolution")
    print("  Mode A: spikes_per_packet=1 (timing-exact, official gate)")
    print("  Mode B: spikes_per_packet=N  (chunked, trade-off report)")
    print("=" * 72)

    test_tonic_format(results)
    test_mode_a_nmnist(results)
    test_mode_a_dvsgesture(results)
    test_mode_b_nmnist(results)
    test_mode_b_dvsgesture(results)
    test_multi_sample(results)
    test_transform_chain(results)

    print("\n" + results.report())
    return results


if __name__ == '__main__':
    r = main()
    sys.exit(0 if r.failed == 0 else 1)
