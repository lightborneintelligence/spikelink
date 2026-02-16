"""
SpikeLink Test Suite
====================

Platform integration tests validating the adapter contract:

    python -m pytest tests/                     # Run all tests
    python tests/test_platform_pynn.py          # PyNN: 25/25
    python tests/test_platform_nest.py          # NEST: 25/25

Test Invariants (enforced across all adapters):
    COUNT     — Exact spike count preservation
    ORDER     — Causal ordering / monotonicity
    TIMING    — 10 μs timing floor
    ROUNDTRIP — Full codec pipeline
    MULTI     — Multi-neuron/sender separation
    EDGE      — Error handling, edge cases
    STRESS    — High-volume validation

Includes nest_mock for CI testing without NEST installation.

Lightborne Intelligence · Dallas TX
"""

# Expose nest_mock for import by test files
from . import nest_mock

__all__ = [
    "nest_mock",
]
