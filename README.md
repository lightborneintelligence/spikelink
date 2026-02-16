SpikeLink

![PyPI](https://img.shields.io/pypi/v/spikelink)
![Python](https://img.shields.io/pypi/pyversions/spikelink)
![License](https://img.shields.io/pypi/l/spikelink)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Spike-native transport for neuromorphic systems.
Move spikes as spikes — preserving event identity, causal ordering, and bounded timing — with measurable degradation under noise.



What SpikeLink Is (and Isn’t)

SpikeLink does not replace digital computation.

It is a transport layer inside digital systems that carries spike symbols as sparse event structures instead of collapsing them into dense sample streams or generic packet payloads.

Core principle: preserve event semantics — count, ordering, and timing budget — end-to-end.


Why SpikeLink?

Traditional Transport	SpikeLink
SPIKE → ADC → BITS → DAC → SPIKE	SPIKE → SPIKELINK → SPIKE
Conversion overhead	Event-native transport
Cliff-edge failure	Graceful degradation
Dense payloads / silence shipped	Sparse event semantics
Implicit timing distortion	Explicit timing bounds

Neuromorphic systems are event-native.
Transport should be event-native too.


Transport Guarantees

SpikeLink is governed by a formal transport contract.

It guarantees:
	•	Event count preservation
	•	Causal ordering monotonicity
	•	Explicit timing bounds
	•	Deterministic reconstruction
	•	Monotonic, measurable degradation under noise

See full specification:

docs/transport_contract.md

SpikeLink defines transport semantics for spike-native systems operating inside conventional digital infrastructure.


Key Properties
	•	Spike-native: no mandatory ADC/DAC conversion stages for transport
	•	Graceful degradation: precision reduces under noise, not event loss
	•	Time-coherent: explicit timing budget, no hidden drift
	•	Deterministic: encode → decode preserves ordering and bounds
	•	Ecosystem-ready: broad adapter coverage across simulators and hardware stacks


Adapter Ecosystem

SpikeLink now supports a full cross-platform adapter layer:


#Adapter	Target Platform	Status

Neo	EBRAINS ecosystem	✅
Brian2	Brian2 simulator	✅
Tonic	Event camera datasets	✅
PyNN	Multi-backend abstraction	✅
NEST	NEST simulator	✅
Nengo	Nengo neuromorphic framework	✅
Lava	Intel Loihi (Lava stack)	✅
SpikeInterface	Electrophysiology / spike sorting	✅

All adapters preserve transport invariants and use seconds as the canonical internal time base.


Validated Platforms

SpikeLink has been validated across independent integration surfaces with invariant enforcement tests.

Core validation surfaces:
	•	Neo / EBRAINS
	•	Brian2
	•	Tonic-format event datasets
	•	PyNN
	•	NEST

Platform tests enforce:
	•	Exact spike/event count preservation
	•	Causal ordering monotonicity
	•	Timing budget compliance
	•	Trade-off transparency (v2 chunk mode)


Timing Model (v2)

SpikeLink v2 encodes absolute time using:

chunk_start_10us

Resolution: 10 μs

Two explicit modes:
	•	Mode A — Timing-Exact: timing preserved within 10 μs quantization floor
	•	Mode B — Chunked: reduced packet count, increased intra-chunk distortion (quantified and reported)

This is a declared trade-off — not hidden behavior.


Install

pip install spikelink

Optional extras:

pip install spikelink[neo]
pip install spikelink[pynn]
pip install spikelink[nest]
pip install spikelink[nengo]
pip install spikelink[lava]
pip install spikelink[spikeinterface]
pip install spikelink[full]

(Some platforms require their native simulator installed separately.)


Quick Start

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


Neo Integration Example

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


Verification & Degradation Profiling

from spikelink import VerificationSuite, DegradationProfiler

suite = VerificationSuite()
results = suite.run_all()
suite.print_results(results)

profiler = DegradationProfiler()
profile = profiler.profile(noise_levels=[0, 0.1, 1.0, 10.0])
profiler.print_profile(profile)

Degradation is designed to be monotonic:
confidence must never inflate under noise.


Repository Structure

'''
spikelink/
├── src/spikelink/
│   ├── api.py
│   ├── types/
│   ├── core/
│   ├── v2/
│   ├── adapters/
│   ├── verification/
│   ├── stress/
│   ├── hw/
│   └── waveml/
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   └── transport_contract.md
├── pyproject.toml
├── README.md
└── LICENSE
'''


Roadmap
	•	Extended channel models (loss, jitter, interference)
	•	Hardware packet handler (FPGA reference path)
	•	Larger-scale event workload benchmarks
	•	Distributed multi-node validation scenarios


License

Apache 2.0 — see LICENSE.


About

Lightborne Intelligence
Truth > Consensus · Sovereignty > Control · Coherence > Speed