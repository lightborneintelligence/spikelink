# SpikeLink

**Spike-native transport for neuromorphic systems.**

[![PyPI](https://img.shields.io/pypi/v/spikelink)](https://pypi.org/project/spikelink/)
[![Python](https://img.shields.io/pypi/pyversions/spikelink)](https://pypi.org/project/spikelink/)
[![License](https://img.shields.io/pypi/l/spikelink)](https://github.com/lightborneintelligence/spikelink/blob/main/LICENSE)
[![Tests](https://github.com/lightborneintelligence/spikelink/actions/workflows/ci.yml/badge.svg)](https://github.com/lightborneintelligence/spikelink/actions)

---

> **Note on representation**  
> SpikeLink does not replace digital computation. It operates *post-binary within digital systems* by transporting spike symbols without forcing early binary collapse.

---

## Why SpikeLink?

| Traditional Transport | SpikeLink |
|-----------------------|-----------|
| SPIKE → ADC → BITS → DAC → SPIKE | SPIKE → SPIKELINK → SPIKE |
| Conversion overhead | Native transport |
| Cliff-edge failure | Graceful degradation |
| Precision loss | Symbol magnitude continuity |

**Key Properties**

- **Spike-native**: No ADC/DAC conversion stages  
- **Graceful degradation**: Precision loss under noise, not data loss  
- **EBRAINS compatible**: Verified against Neo, Elephant, PyNN workflows  
- **Time-coherent**: Bounded timing, predictable behavior under fixed conditions  

---

## Install

```bash
pip install spikelink
````

With EBRAINS ecosystem support:

```bash
pip install spikelink[neo]      # Neo adapter
pip install spikelink[elephant] # Neo + Elephant
pip install spikelink[full]     # Everything
```

---

## Quick Start

```python
from spikelink import SpikeTrain, SpikelinkCodec

# Create a spike train
train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])

# Encode to packets
codec = SpikelinkCodec()
packets = codec.encode_train(train)

# Decode back
recovered = codec.decode_packets(packets)

print(f"Original:  {train.times}")
print(f"Recovered: {recovered.times}")
```

---

### Convenience API

```python
import spikelink

original = [0.1, 0.2, 0.3, 0.4, 0.5]

packets = spikelink.encode(original)
recovered = spikelink.decode(packets)

passed = spikelink.verify(original, recovered)
print("Verification:", "PASS" if passed else "FAIL")
```

---

### Neo Integration

```python
from spikelink.adapters import NeoAdapter
from spikelink import SpikelinkCodec
import neo
import quantities as pq

# From Neo SpikeTrain
neo_train = neo.SpikeTrain([0.1, 0.2, 0.3] * pq.s, t_stop=1.0 * pq.s)
our_train = NeoAdapter.from_neo(neo_train)

# Transport through SpikeLink
codec = SpikelinkCodec()
packets = codec.encode_train(our_train)
recovered = codec.decode_packets(packets)

# Back to Neo
recovered_neo = NeoAdapter.to_neo(recovered)
```

---

## Verification

```python
from spikelink import VerificationSuite, DegradationProfiler

suite = VerificationSuite()
results = suite.run_all(train)
suite.print_results(results)

profiler = DegradationProfiler()
profile = profiler.profile(train, noise_levels=[0, 0.1, 1.0, 10.0])
profiler.print_profile(profile)
```

---

## Graceful Degradation

SpikeLink degrades proportionally under noise — **precision loss, not data loss**:

| Noise % | Decimals Preserved |
| ------- | ------------------ |
| 0.0%    | 4                  |
| 0.1%    | 3                  |
| 1.0%    | 2                  |
| 10.0%   | 1                  |

✓ Monotonic degradation confirmed (confidence never inflates)

---

## Documentation

* [Full Documentation](https://spikelink.readthedocs.io)
* [Protocol Specification](https://spikelink.readthedocs.io/specifications/protocol/)
* [EBRAINS Integration Guide](https://spikelink.readthedocs.io/tutorials/neo-integration/)

---

## External Verification

SpikeLink has been externally verified against EBRAINS workflows:

* **Neo-compatible**: SpikeTrain round-trip verified
* **Elephant-verified**: Statistical fidelity confirmed
* **PyNN stress-tested**: 380 neurons, 13,344 spikes, 100% symbol preservation
* **Ready for integration with SpiNNaker and BrainScaleS**

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## About

**Lightborne Intelligence**

*Truth > Consensus · Sovereignty > Control · Coherence > Speed*
