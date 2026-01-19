# Changelog

All notable changes to SpikeLink will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-17
## Draft Release Notes — v0.1.2

These are documentation-only patch notes, exactly what reviewers expect.

### Added

- Initial release
- Core `SpikeTrain` data type
- `SpikelinkCodec` for encoding/decoding spike trains to 32-byte packets
- `SpikelinkPacket` wire format (7 spikes per packet)
- `TransportSimulator` for noise injection testing
- `NeoAdapter` for EBRAINS ecosystem interoperability
- `VerificationSuite` with 6-axis validation
- `DegradationProfiler` for graceful degradation analysis
- `StatisticalAnalysis` with Elephant-compatible metrics
- `PyNNStyleGenerator` for stress testing
- Convenience API (`encode`, `decode`, `verify`)

### Validation

- Neo round-trip compatibility verified
- Elephant statistical fidelity confirmed
- Graceful degradation monotonicity proven
- PyNN-style population stress tests passed (100+ neurons)

---

**Lightborne Intelligence** 

— *Truth > Consensus · Sovereignty > Control · Coherence > Speed*
