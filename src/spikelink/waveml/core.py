"""
WaveML Core — Wave-Native Primitives for SpikeLink v2
=====================================================
Lightborne Intelligence

Extracted and hardened WaveML functions for transport integration:
- WaveState: amplitude + phase representation
- HarmonicTransform: Chebyshev-grid shell decomposition
- ERA: Shell-aware error regulation
- ShellMap: Semantic shell allocation

Truth > Consensus. Sovereignty > Control. Coherence > Speed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import IntEnum

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
PI = np.pi


# =============================================================================
# SHELL TIERS — Semantic classification
# =============================================================================

class ShellTier(IntEnum):
    """
    Shell semantic tiers.
    
    Identity:  The signal IS this. Tight bounds.
    Structure: The signal's shape. Moderate bounds.
    Dynamics:  How the signal moves. Loose bounds.
    Noise:     Not the signal. Capped/discardable.
    """
    IDENTITY  = 0
    STRUCTURE = 1
    DYNAMICS  = 2
    NOISE     = 3


@dataclass
class ShellMap:
    """
    Maps shell indices to semantic tiers.
    
    For SpikeLink v2 with 7 bands:
      Bands 0-1: Identity  (tight — high precision allocation)
      Bands 2-3: Structure (moderate)
      Bands 4-5: Dynamics  (loose — tolerate degradation)
      Band  6:   Noise     (capped — first to degrade)
    """
    n_shells: int = 7
    identity_end: int = 2      # [0, identity_end)
    structure_end: int = 4     # [identity_end, structure_end)
    dynamics_end: int = 6      # [structure_end, dynamics_end)
    # Remaining shells are noise: [dynamics_end, n_shells)
    
    def tier(self, shell_idx: int) -> ShellTier:
        if shell_idx < self.identity_end:
            return ShellTier.IDENTITY
        elif shell_idx < self.structure_end:
            return ShellTier.STRUCTURE
        elif shell_idx < self.dynamics_end:
            return ShellTier.DYNAMICS
        else:
            return ShellTier.NOISE
    
    def tier_mask(self, tier: ShellTier) -> np.ndarray:
        """Boolean mask for shells in given tier."""
        mask = np.zeros(self.n_shells, dtype=bool)
        for i in range(self.n_shells):
            if self.tier(i) == tier:
                mask[i] = True
        return mask
    
    def precision_weights(self) -> np.ndarray:
        """
        Precision allocation weights per shell.
        Identity gets the most bits, noise gets the least.
        """
        weights = np.zeros(self.n_shells)
        for i in range(self.n_shells):
            tier = self.tier(i)
            if tier == ShellTier.IDENTITY:
                weights[i] = 1.0
            elif tier == ShellTier.STRUCTURE:
                weights[i] = 0.7
            elif tier == ShellTier.DYNAMICS:
                weights[i] = 0.4
            else:
                weights[i] = 0.15
        return weights


# =============================================================================
# WAVE STATE
# =============================================================================

@dataclass
class WaveState:
    """
    Wave-native state: amplitude + phase per shell.
    
    This is the canonical representation inside WaveML.
    SpikeLink v2 transports WaveStates instead of raw values.
    """
    amplitude: np.ndarray   # (K,) shell magnitudes, non-negative
    phase: np.ndarray       # (K,) shell phases in [-π, π]
    
    @property
    def n_modes(self) -> int:
        return len(self.amplitude)
    
    @property
    def energy(self) -> float:
        return float(np.sum(self.amplitude ** 2))
    
    @property
    def shell_energy(self) -> np.ndarray:
        return self.amplitude ** 2
    
    def complex_form(self) -> np.ndarray:
        """Convert to complex representation: A * exp(j*φ)"""
        return self.amplitude * np.exp(1j * self.phase)
    
    @staticmethod
    def from_complex(z: np.ndarray) -> 'WaveState':
        """Construct from complex representation."""
        return WaveState(
            amplitude=np.abs(z),
            phase=np.angle(z)
        )
    
    def copy(self) -> 'WaveState':
        return WaveState(
            amplitude=self.amplitude.copy(),
            phase=self.phase.copy()
        )


# =============================================================================
# HARMONIC TRANSFORM
# =============================================================================

class HarmonicTransform:
    """
    Chebyshev-grid Harmonic Transform.
    
    Unlike FFT:
    - Respects finite signal boundaries (no leakage)
    - Chebyshev nodes for optimal conditioning
    - Curvature-weighted shells (physical scaling)
    - O(NK) with cached basis
    
    For spike trains: K = 7 (matching SpikeLink bands)
    """
    
    def __init__(self, n_shells: int = 7, curvature_scale: float = 1.1):
        self.K = n_shells
        self.lam = curvature_scale  # λ for curvature weighting
        self._basis_cache = {}
    
    def _chebyshev_nodes(self, N: int) -> np.ndarray:
        """Chebyshev nodes on [-1, 1]."""
        i = np.arange(N)
        return np.cos(PI * (2 * i + 1) / (2 * N))
    
    def _build_basis(self, N: int) -> np.ndarray:
        """
        Build Chebyshev basis matrix (N x K).
        
        Cached per signal length for efficiency.
        """
        if N in self._basis_cache:
            return self._basis_cache[N]
        
        x = self._chebyshev_nodes(N)
        B = np.zeros((N, self.K))
        
        for k in range(self.K):
            # Chebyshev polynomial T_k evaluated at nodes
            B[:, k] = np.cos(k * np.arccos(np.clip(x, -1, 1)))
            # Curvature weighting: higher shells get λ^k scaling
            B[:, k] *= self.lam ** (-k)
        
        self._basis_cache[N] = B
        return B
    
    def forward(self, signal: np.ndarray) -> WaveState:
        """
        Signal → WaveState decomposition.
        
        Projects signal onto Chebyshev shells, returns
        amplitude and phase per shell.
        """
        N = len(signal)
        B = self._build_basis(N)
        
        # Least-squares projection: coefficients = pinv(B) @ signal
        coeffs = np.linalg.lstsq(B, signal, rcond=None)[0]
        
        # Convert real coefficients to amplitude + phase
        # For real signals: amplitude = |coeff|, phase = 0 or π
        amplitude = np.abs(coeffs)
        phase = np.where(coeffs >= 0, 0.0, PI)
        
        return WaveState(amplitude=amplitude, phase=phase)
    
    def forward_complex(self, signal: np.ndarray) -> WaveState:
        """
        Complex signal → WaveState decomposition.
        
        For spike trains with timing (phase) information.
        """
        N = len(signal)
        B = self._build_basis(N)
        
        # Complex projection
        coeffs_real = np.linalg.lstsq(B, np.real(signal), rcond=None)[0]
        coeffs_imag = np.linalg.lstsq(B, np.imag(signal), rcond=None)[0]
        
        coeffs = coeffs_real + 1j * coeffs_imag
        
        return WaveState(
            amplitude=np.abs(coeffs),
            phase=np.angle(coeffs)
        )
    
    def inverse(self, state: WaveState, N: int) -> np.ndarray:
        """
        WaveState → Signal reconstruction.
        """
        B = self._build_basis(N)
        
        # Reconstruct coefficients from amplitude + phase
        coeffs = state.amplitude * np.cos(state.phase)
        
        return B @ coeffs
    
    def inverse_complex(self, state: WaveState, N: int) -> np.ndarray:
        """
        WaveState → Complex signal reconstruction.
        """
        B = self._build_basis(N)
        coeffs = state.complex_form()
        return B @ np.real(coeffs) + 1j * (B @ np.imag(coeffs))


# =============================================================================
# ERA — ERROR REGULATION ARCHITECTURE
# =============================================================================

@dataclass
class ERABounds:
    """
    Shell-aware ERA bounds for transport.
    
    Each tier has different tolerance for amplitude drift,
    energy accumulation, and phase rotation.
    """
    # Identity tier (very tight)
    identity_max_amplitude: float = 2.0
    identity_max_energy: float = 10.0
    identity_max_phase_drift: float = 0.1  # radians
    
    # Structure tier (tight)
    structure_max_amplitude: float = 1.5
    structure_max_energy: float = 8.0
    structure_max_phase_drift: float = 0.3
    
    # Dynamics tier (moderate)
    dynamics_max_amplitude: float = 1.0
    dynamics_max_energy: float = 5.0
    dynamics_max_phase_drift: float = 0.8
    
    # Noise tier (capped)
    noise_max_amplitude: float = 0.3
    noise_max_energy: float = 1.0
    noise_max_phase_drift: float = PI  # any phase ok
    
    def bounds_for_tier(self, tier: ShellTier) -> Tuple[float, float, float]:
        """Return (max_amp, max_energy, max_phase_drift) for tier."""
        if tier == ShellTier.IDENTITY:
            return (self.identity_max_amplitude,
                    self.identity_max_energy,
                    self.identity_max_phase_drift)
        elif tier == ShellTier.STRUCTURE:
            return (self.structure_max_amplitude,
                    self.structure_max_energy,
                    self.structure_max_phase_drift)
        elif tier == ShellTier.DYNAMICS:
            return (self.dynamics_max_amplitude,
                    self.dynamics_max_energy,
                    self.dynamics_max_phase_drift)
        else:
            return (self.noise_max_amplitude,
                    self.noise_max_energy,
                    self.noise_max_phase_drift)


class ERA:
    """
    Error Regulation Architecture — Transport Edition.
    
    Applies shell-aware bounds at encode/decode boundaries.
    
    Design rule:
        ERA guards meaning, not representation.
        HT removes the representation burden before ERA sees it.
    
    For SpikeLink v2:
    - Pre-encode ERA: protect identity before transport compression
    - Post-decode ERA: correct transport-induced drift
    - Curvature-weighted correction at high-curvature moments
    """
    
    def __init__(self, 
                 shell_map: ShellMap,
                 bounds: Optional[ERABounds] = None):
        self.shell_map = shell_map
        self.bounds = bounds or ERABounds()
        self._prev_state: Optional[WaveState] = None
    
    def rectify(self, state: WaveState, 
                prev_state: Optional[WaveState] = None) -> WaveState:
        """
        Full ERA rectification pipeline:
        
        1. Non-negativity (amplitude ≥ 0)
        2. Shell-specific amplitude clamping
        3. Shell-specific energy bounds
        4. Phase wrapping to [-π, π]
        5. Phase drift gating (if previous state provided)
        """
        amp = state.amplitude.copy()
        phase = state.phase.copy()
        
        prev = prev_state or self._prev_state
        
        for i in range(state.n_modes):
            tier = self.shell_map.tier(i)
            max_amp, max_energy, max_phase_drift = self.bounds.bounds_for_tier(tier)
            
            # Step 1: Non-negativity
            amp[i] = max(amp[i], 0.0)
            
            # Step 2: Amplitude clamping
            amp[i] = min(amp[i], max_amp)
            
            # Step 3: Energy bounds (per-shell)
            if amp[i] ** 2 > max_energy:
                amp[i] = np.sqrt(max_energy)
            
            # Step 4: Phase wrapping
            phase[i] = ((phase[i] + PI) % (2 * PI)) - PI
            
            # Step 5: Phase drift gating
            if prev is not None and i < prev.n_modes:
                drift = abs(phase[i] - prev.phase[i])
                if drift > PI:
                    drift = 2 * PI - drift  # unwrap
                if drift > max_phase_drift:
                    # Snap back toward previous phase
                    direction = np.sign(phase[i] - prev.phase[i])
                    phase[i] = prev.phase[i] + direction * max_phase_drift
                    phase[i] = ((phase[i] + PI) % (2 * PI)) - PI
        
        result = WaveState(amplitude=amp, phase=phase)
        self._prev_state = result.copy()
        return result
    
    def curvature_weight(self, state: WaveState, 
                         prev_state: Optional[WaveState] = None,
                         prev_prev_state: Optional[WaveState] = None
                         ) -> np.ndarray:
        """
        Compute curvature weight per shell.
        
        High curvature → more correction needed
        Low curvature → trust the observation
        
        κ(t) ≈ |d²y/dt²| estimated from three consecutive states.
        """
        if prev_state is None or prev_prev_state is None:
            return np.ones(state.n_modes)
        
        # Second derivative estimate per shell
        d2 = (state.amplitude - 2 * prev_state.amplitude + 
              prev_prev_state.amplitude)
        
        kappa = np.abs(d2)
        kappa_max = np.max(kappa) + 1e-12
        
        # Normalized to [0, 1] with tanh smoothing
        weights = np.tanh(kappa / kappa_max)
        
        return weights
    
    def adaptive_rectify(self, state: WaveState,
                         prev_state: Optional[WaveState] = None,
                         prev_prev_state: Optional[WaveState] = None
                         ) -> WaveState:
        """
        Curvature-weighted ERA rectification.
        
        At high-curvature moments (transitions, peaks), correction
        is stronger to protect identity. At flat moments, observation
        is trusted more.
        """
        # First: standard rectification
        rectified = self.rectify(state, prev_state)
        
        if prev_state is None:
            return rectified
        
        # Compute curvature weights
        weights = self.curvature_weight(state, prev_state, prev_prev_state)
        
        # Blend: more weight → closer to rectified (protected)
        # Less weight → closer to original (trusted)
        amp_blended = (weights * rectified.amplitude + 
                       (1 - weights) * state.amplitude)
        phase_blended = rectified.phase  # phase always fully rectified
        
        return WaveState(amplitude=amp_blended, phase=phase_blended)
    
    def reset(self):
        """Reset drift tracking state."""
        self._prev_state = None


# =============================================================================
# CONVENIENCE
# =============================================================================

# Default configurations
DEFAULT_SHELL_MAP = ShellMap(n_shells=7)
DEFAULT_HT = HarmonicTransform(n_shells=7)
DEFAULT_ERA_BOUNDS = ERABounds()
