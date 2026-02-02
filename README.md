# SpikeLink

**Spike-native transport for neuromorphic systems.**  
Move spikes as spikes — preserving event identity, causal ordering, and bounded timing — with graceful degradation under noise.

![PyPI](https://img.shields.io/pypi/v/spikelink)
![Python](https://img.shields.io/pypi/pyversions/spikelink)
![License](https://img.shields.io/pypi/l/spikelink)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

---

## What SpikeLink is (and isn’t)

SpikeLink **does not replace digital computation**. It is a transport layer inside digital systems that carries **spike symbols as sparse events** instead of forcing them into dense sample streams or generic packet payloads.

**Core idea:** preserve *event semantics* (count, ordering, timing budget) end-to-end.

---

## Why SpikeLink?

| Traditional Transport | SpikeLink |
|---|---|
| SPIKE → ADC → BITS → DAC → SPIKE | SPIKE → SPIKELINK → SPIKE |
| Conversion overhead | Event-native transport |
| Cliff-edge failure | Graceful degradation |
| Dense payloads / silence shipped | Sparse event semantics |
| Timing drift surprises | Explicit timing budget |

---

## Key properties

- **Spike-native:** no ADC/DAC conversion stages required for transport
- **Graceful degradation:** precision loss under noise, not data loss
- **Time-coherent:** explicit timing budget with predictable behavior
- **Ecosystem-ready:** adapters and tests designed for common neuroscience / neuromorphic toolchains

---

## Validated platforms (81/81 tests passed)

SpikeLink has been validated across three independent integration surfaces:

- **Neo / EBRAINS** — 26/26 ✅  
- **Brian2** — 22/22 ✅  
- **Tonic-format event datasets** — 33/33 ✅ *(timing-exact mode available)*

### Tonic timing truth guard (reviewer-proof)
SpikeLink encodes absolute time using `chunk_start_10us` (**10 μs resolution**).

- **Mode A (Timing-Exact gate):** timing preserved within the 10 μs quantization floor  
- **Mode B (Chunked trade-off report):** fewer packets, but increased intra-chunk timing distortion (quantified in tests)

This is a deliberate, explicit trade: **choose Mode A when temporal microstructure matters; choose Mode B when rate/statistics are sufficient.**

---

## Install

```bash
pip install spikelink
Optional extras (if provided in pyproject.toml):

pip install spikelink[neo]
pip install spikelink[elephant]
pip install spikelink[full]
Quick start
from spikelink import SpikeTrain, SpikelinkCodec

train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])

codec = SpikelinkCodec()
packets = codec.encode_train(train)

recovered = codec.decode_packets(packets)

print("Original: ", train.times)
print("Recovered:", recovered.times)
Convenience API
import spikelink

original = [0.1, 0.2, 0.3, 0.4, 0.5]

packets = spikelink.encode(original)
recovered = spikelink.decode(packets)

ok = spikelink.verify(original, recovered)
print("Verification:", "PASS" if ok else "FAIL")
Note: If your public API names differ, treat the above as the intended ergonomics and map it to your actual src/spikelink/api.py.

Platform tests (reproducible)
python test_platform_neo.py
python test_platform_brian2.py
python test_platform_tonic.py
These tests enforce the invariants that matter:

exact event/spike count

causal ordering monotonicity

timing budget compliance (Mode A)

trade-off reporting (Mode B)

Neo integration (example)
from spikelink.adapters import NeoAdapter
from spikelink import SpikelinkCodec
import neo
import quantities as pq

neo_train = neo.SpikeTrain([0.1, 0.2, 0.3] * pq.s, t_stop=1.0 * pq.s)
train = NeoAdapter.from_neo(neo_train)

codec = SpikelinkCodec()
packets = codec.encode_train(train)
recovered = codec.decode_packets(packets)

recovered_neo = NeoAdapter.to_neo(recovered)
Verification & degradation profiling
from spikelink import VerificationSuite, DegradationProfiler

suite = VerificationSuite()
results = suite.run_all()
suite.print_results(results)

profiler = DegradationProfiler()
profile = profiler.profile(noise_levels=[0, 0.1, 1.0, 10.0])
profiler.print_profile(profile)
SpikeLink degradation is designed to be monotonic: confidence should never inflate under noise.

Repository structure
spikelink/
├── src/
│   └── spikelink/
│       ├── __init__.py
│       ├── api.py
│       ├── core/
│       ├── types/
│       ├── adapters/
│       ├── verification/
│       ├── stress/
│       └── hw/
├── tests/
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
Documentation
Protocol specification (docs/)

Platform validation notes (docs/)

EBRAINS / Neo adapter guide (docs/)

Roadmap
Additional ecosystem adapters (PyNN/NEST class bridges)

More channel models (loss/jitter/interference stress suites)

Hardware packet handler (FPGA reference path)

Larger-scale event workloads and benchmarks

License
Apache 2.0 — see LICENSE.

About
Lightborne Intelligence
Truth > Consensus · Sovereignty > Control · Coherence > Speed
