```
═══════════════════════════════════════════════════════════════
 ███████╗██████╗ ██╗██╗  ██╗███████╗██╗     ██╗███╗   ██╗██╗  ██╗
 ██╔════╝██╔══██╗██║██║ ██╔╝██╔════╝██║     ██║████╗  ██║██║ ██╔╝
 ███████╗██████╔╝██║█████╔╝ █████╗  ██║     ██║██╔██╗ ██║█████╔╝
 ╚════██║██╔═══╝ ██║██╔═██╗ ██╔══╝  ██║     ██║██║╚██╗██║██╔═██╗
 ███████║██║     ██║██║  ██╗███████╗███████╗██║██║ ╚████║██║  ██╗
 ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝

             Spike-Native Transport Protocol v2
═══════════════════════════════════════════════════════════════
```
![PyPI](https://img.shields.io/pypi/v/spikelink)
![Python](https://img.shields.io/pypi/pyversions/spikelink)
![License](https://img.shields.io/pypi/l/spikelink)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

---

## Event-Native Transport for Neuromorphic Systems

Move spikes as spikes — preserving event identity, causal ordering, and bounded timing — with measurable degradation under noise.

SpikeLink defines a **formal transport contract** for spike-native computation operating inside conventional digital infrastructure.

---

# The Problem

Neuromorphic processors (Loihi, Akida, SpiNNaker, etc.) compute using:

* Asynchronous spikes
* Temporal coding
* Sparse event streams

The moment spikes leave the chip, they are typically forced through:

```
SPIKE → ADC → BITS → PACKET → BITS → DAC → SPIKE
```

This collapses temporal semantics, introduces distortion, and creates cliff-edge failure modes.

---

# SpikeLink Approach

```
SPIKE → SPIKELINK → SPIKE
```

No forced analog collapse.
No hidden timing drift.
No semantic destruction.

Transport preserves event structure.

---

# Chip-to-Chip Transport Model

```
┌───────────────┐         ┌────────────────┐         ┌───────────────┐
│ Neuromorphic  │  SPIKE  │  SpikeLink     │  SPIKE  │ Neuromorphic  │
│ Core (A)      ├────────►│  Encoder       ├────────►│ Core (B)      │
│               │         │  + Contract    │         │               │
└───────────────┘         └────────────────┘         └───────────────┘
                                  │
                                  ▼
                         Bounded timing
                         Causal monotonicity
                         Count preservation
```

SpikeLink packets carry **event identity**, not reconstructed waveforms.

---

# Transport Contract

SpikeLink enforces invariants:

* Event count preservation
* Causal ordering monotonicity
* Explicit timing bounds
* Deterministic reconstruction
* Monotonic degradation under noise

Formal specification:

```
docs/transport_contract.md
```

This is a declared contract — not a best-effort approximation.

---

# Timing Model (v2)

Absolute time encoding:

```
chunk_start_10us
```

Resolution: **10 μs**

Two modes:

| Mode         | Behavior                                      |
| ------------ | --------------------------------------------- |
| Timing-Exact | Preserves timing within quantization floor    |
| Chunked      | Fewer packets, bounded intra-chunk distortion |

Trade-offs are explicit and measurable.

---

# Architecture Overview

```
spikelink/
├── api.py                → Public interface
├── core/                 → Codec + invariants
├── v2/                   → Timing engine
├── types/                → SpikeTrain + packet structs
├── adapters/             → Neo, PyNN, NEST, Lava, etc.
├── verification/         → Contract enforcement suite
├── stress/               → Noise injection + degradation tests
├── hw/                   → Hardware reference hooks
└── waveml/               → Wave-native integration layer
```

Verification and degradation modeling are first-class components.

---

# Adapter Ecosystem

| Adapter        | Target            | Status |
| -------------- | ----------------- | ------ |
| Neo            | EBRAINS           | ✅      |
| Brian2         | Simulator         | ✅      |
| Tonic          | Event datasets    | ✅      |
| PyNN           | Multi-backend     | ✅      |
| NEST           | Simulator         | ✅      |
| Nengo          | Framework         | ✅      |
| Lava           | Intel Loihi       | ✅      |
| SpikeInterface | Electrophysiology | ✅      |

All adapters use **seconds** as canonical time base.

---

# Install

```bash
pip install spikelink
```

Optional integrations:

```bash
pip install spikelink[neo]
pip install spikelink[pynn]
pip install spikelink[nest]
pip install spikelink[nengo]
pip install spikelink[lava]
pip install spikelink[spikeinterface]
pip install spikelink[full]
```

---

# Quick Start

```python
from spikelink import SpikeTrain, SpikelinkCodec

train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])

codec = SpikelinkCodec()
packets = codec.encode_train(train)
recovered = codec.decode_packets(packets)

print(train.times)
print(recovered.times)
```

---

# Verification & Degradation Profiling

```python
from spikelink import VerificationSuite, DegradationProfiler

suite = VerificationSuite()
suite.print_results(suite.run_all())

profiler = DegradationProfiler()
profile = profiler.profile(noise_levels=[0, 0.1, 1.0, 10.0])
profiler.print_profile(profile)
```

Degradation must be monotonic.
Confidence must never inflate under noise.

---

# Design Philosophy

SpikeLink does not compete with digital systems.
It restores event semantics inside them.

Neuromorphic systems are event-native.
Transport must be event-native.

---

# Roadmap

* Extended channel models (loss, jitter, interference)
* FPGA packet handler reference path
* Distributed multi-node validation
* Cross-chip hardware benchmarks

---

# License

Apache 2.0

---

# About

Lightborne Intelligence

Event semantics over collapse.
Coherence over distortion.
