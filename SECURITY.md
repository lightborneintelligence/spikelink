# Security Policy

## Trust Boundary

The public `spikelink` package provides interfaces and reference checks sufficient for:

- **Interoperability**: Integration with EBRAINS ecosystem (Neo, Elephant, PyNN)
- **Correctness verification**: Round-trip validation and statistical fidelity tests
- **Performance characterization**: Graceful degradation profiling

Extended validation artifacts, proprietary optimizations, and internal specifications are maintained separately under controlled access.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in SpikeLink, please report it responsibly:

1. **Do not** open a public issue
2. Email security@lightborneintelligence.com with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Scope

This security policy covers:

- The `spikelink` Python package
- The packet encoding/decoding protocol
- Integration adapters (Neo, etc.)

This policy does not cover:

- Third-party dependencies (Neo, Elephant, numpy)
- Hardware implementations
- Extended verification artifacts

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve SpikeLink.

---

**Lightborne Intelligence** — *Truth > Consensus · Sovereignty > Control · Coherence > Speed*
