"""
VerificationSuite — Validate SpikeLink protocol correctness.

Tests round-trip fidelity, timing preservation, and statistical properties
to ensure the protocol behaves correctly under various conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from spikelink.types.spiketrain import SpikeTrain
from spikelink.core.codec import SpikelinkCodec


@dataclass
class VerificationResult:
    """Result of a single verification test."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    results: List[VerificationResult]
    
    @property
    def passed(self) -> bool:
        """True if all tests passed."""
        return all(r.passed for r in self.results)
    
    @property
    def n_passed(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r.passed)
    
    @property
    def n_failed(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.results if not r.passed)
    
    def summary(self) -> str:
        """Return a summary string."""
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: {self.n_passed}/{len(self.results)} tests passed"


class VerificationSuite:
    """
    Suite of verification tests for SpikeLink protocol.
    
    Tests include:
    - Round-trip fidelity (encode-decode preserves data)
    - Timing bounds (no spikes outside window)
    - Ordering preservation (spikes remain sorted)
    - Statistical fidelity (ISI distribution preserved)
    
    Example:
        >>> suite = VerificationSuite()
        >>> train = SpikeTrain(times=[0.1, 0.2, 0.3, 0.4, 0.5])
        >>> report = suite.run_all(train)
        >>> print(report.summary())
        PASSED: 5/5 tests passed
    """
    
    def __init__(self, codec: Optional[SpikelinkCodec] = None):
        """
        Initialize verification suite.
        
        Args:
            codec: Codec to test (default: standard SpikelinkCodec)
        """
        self.codec = codec or SpikelinkCodec()
    
    def verify_round_trip(
        self,
        train: SpikeTrain,
        tolerance: float = 1e-6,
    ) -> VerificationResult:
        """
        Verify that encode-decode round trip preserves spike times.
        
        Args:
            train: Original spike train
            tolerance: Maximum allowed error per spike
            
        Returns:
            VerificationResult
        """
        recovered = self.codec.round_trip(train)
        
        if len(train) != len(recovered):
            return VerificationResult(
                name="round_trip",
                passed=False,
                message=f"Spike count mismatch: {len(train)} -> {len(recovered)}",
            )
        
        if len(train) == 0:
            return VerificationResult(
                name="round_trip",
                passed=True,
                message="Empty train preserved",
            )
        
        max_error = np.max(np.abs(train.times - recovered.times))
        passed = max_error <= tolerance
        
        return VerificationResult(
            name="round_trip",
            passed=passed,
            message=f"Max error: {max_error:.2e} (tolerance: {tolerance:.2e})",
            details={"max_error": float(max_error), "tolerance": tolerance},
        )
    
    def verify_timing_bounds(self, train: SpikeTrain) -> VerificationResult:
        """
        Verify that recovered spikes stay within timing bounds.
        
        Args:
            train: Original spike train
            
        Returns:
            VerificationResult
        """
        recovered = self.codec.round_trip(train)
        
        if len(recovered) == 0:
            return VerificationResult(
                name="timing_bounds",
                passed=True,
                message="Empty train (trivially valid)",
            )
        
        min_time = np.min(recovered.times)
        max_time = np.max(recovered.times)
        
        in_bounds = (min_time >= recovered.t_start) and (max_time <= recovered.t_stop)
        
        return VerificationResult(
            name="timing_bounds",
            passed=in_bounds,
            message=f"Times in [{min_time:.6f}, {max_time:.6f}], bounds [{recovered.t_start}, {recovered.t_stop}]",
            details={
                "min_time": float(min_time),
                "max_time": float(max_time),
                "t_start": recovered.t_start,
                "t_stop": recovered.t_stop,
            },
        )
    
    def verify_ordering(self, train: SpikeTrain) -> VerificationResult:
        """
        Verify that spike ordering is preserved.
        
        Args:
            train: Original spike train
            
        Returns:
            VerificationResult
        """
        recovered = self.codec.round_trip(train)
        
        if len(recovered) < 2:
            return VerificationResult(
                name="ordering",
                passed=True,
                message="Fewer than 2 spikes (trivially ordered)",
            )
        
        is_sorted = np.all(np.diff(recovered.times) >= 0)
        
        return VerificationResult(
            name="ordering",
            passed=is_sorted,
            message="Spikes are sorted" if is_sorted else "Spikes are NOT sorted",
        )
    
    def verify_spike_count(self, train: SpikeTrain) -> VerificationResult:
        """
        Verify that spike count is preserved.
        
        Args:
            train: Original spike train
            
        Returns:
            VerificationResult
        """
        recovered = self.codec.round_trip(train)
        
        passed = len(train) == len(recovered)
        
        return VerificationResult(
            name="spike_count",
            passed=passed,
            message=f"Original: {len(train)}, Recovered: {len(recovered)}",
            details={"original": len(train), "recovered": len(recovered)},
        )
    
    def verify_isi_distribution(
        self,
        train: SpikeTrain,
        tolerance: float = 0.01,
    ) -> VerificationResult:
        """
        Verify that inter-spike interval distribution is preserved.
        
        Args:
            train: Original spike train
            tolerance: Maximum relative error in mean ISI
            
        Returns:
            VerificationResult
        """
        if len(train) < 2:
            return VerificationResult(
                name="isi_distribution",
                passed=True,
                message="Fewer than 2 spikes (no ISI to verify)",
            )
        
        recovered = self.codec.round_trip(train)
        
        original_isi = np.diff(train.times)
        recovered_isi = np.diff(recovered.times)
        
        original_mean = np.mean(original_isi)
        recovered_mean = np.mean(recovered_isi)
        
        if original_mean == 0:
            relative_error = 0.0 if recovered_mean == 0 else float('inf')
        else:
            relative_error = abs(recovered_mean - original_mean) / original_mean
        
        passed = relative_error <= tolerance
        
        return VerificationResult(
            name="isi_distribution",
            passed=passed,
            message=f"Mean ISI: {original_mean:.6f} -> {recovered_mean:.6f} (error: {relative_error:.2%})",
            details={
                "original_mean_isi": float(original_mean),
                "recovered_mean_isi": float(recovered_mean),
                "relative_error": float(relative_error),
            },
        )
    
    def run_all(self, train: SpikeTrain) -> VerificationReport:
        """
        Run all verification tests.
        
        Args:
            train: Spike train to verify
            
        Returns:
            VerificationReport with all results
        """
        results = [
            self.verify_round_trip(train),
            self.verify_timing_bounds(train),
            self.verify_ordering(train),
            self.verify_spike_count(train),
            self.verify_isi_distribution(train),
        ]
        
        return VerificationReport(results=results)
    
    def print_results(self, report: Optional[VerificationReport] = None, train: Optional[SpikeTrain] = None):
        """
        Print verification results in a readable format.
        
        Args:
            report: Pre-computed report (or None to compute from train)
            train: Spike train to verify (if report not provided)
        """
        if report is None:
            if train is None:
                raise ValueError("Must provide either report or train")
            report = self.run_all(train)
        
        print("=" * 50)
        print("SpikeLink Verification Report")
        print("=" * 50)
        
        for result in report.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status}  {result.name}: {result.message}")
        
        print("-" * 50)
        print(report.summary())
        print("=" * 50)
