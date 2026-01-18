# Changelog

All notable changes to SpikeLink will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-17

### Added

- Initial release of SpikeLink
- Core `SpikeTrain` data type for spike sequences
- `SpikelinkCodec` for encoding/decoding spike trains to packets
- `SpikelinkPacket` wire format with bounded precision transport
- `VerificationSuite` for protocol correctness validation
- `DegradationProfiler` for noise characterization
- `NeoAdapter` for EBRAINS ecosystem integration
- Convenience API: `encode()`, `decode()`, `verify()`
- Stress test generators: `generate_population()`, `generate_burst()`, `generate_regular()`
- Comprehensive test suite (unit + integration)
- CI/CD pipeline with GitHub Actions
- PyPI publishing via Trusted Publishers

### Key Properties

- Spike-native transport (no ADC/DAC conversion)
- Graceful degradation under noise (precision loss, not data loss)
- EBRAINS compatible (Neo, Elephant, PyNN validated)
- Time-coherent with bounded timing behavior

---

**Lightborne Intelligence**  
*Truth > Consensus · Sovereignty > Control · Coherence > Speed*
