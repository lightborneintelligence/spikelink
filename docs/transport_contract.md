SpikeLink Transport Contract v1.0

Event-Native Transport for Neuromorphic Systems

1. Scope

SpikeLink defines a transport-layer contract for spike/event-native computation inside digital systems.

It governs how discrete spike events are encoded, transmitted, and reconstructed while preserving:
	•	Event identity
	•	Causal ordering
	•	Bounded timing semantics
	•	Graceful degradation under noise

It does not define inference, learning, or neuron models.
It defines transport guarantees.

2. Canonical Event Model

A spike event is defined as:

E = (t, neuron_id, metadata?)

Where:
	•	t ∈ ℝ⁺ (seconds)
	•	neuron_id ∈ ℕ ∪ Σ
	•	Events are strictly ordered by time per sender
	•	Multiple senders allowed

Time is always represented in seconds internally.

3. Core Invariants

3.1 Event Count Preservation

For a given sender:

|E_in| = |E_out|

No silent spike dropping.
No artificial spike creation.

Exception: explicit degradation modes must be declared and measurable.


3.2 Causal Monotonicity

For all events i, j from the same sender:

if t_i < t_j  →  t_i_out ≤ t_j_out

No time reversal.

Transport may introduce bounded quantization but not causal inversion.


3.3 Timing Bound Guarantee

SpikeLink defines a quantization floor:
	•	v1 codec: float32 precision (~1e-6 sec typical)
	•	v2 codec: 10 µs chunk resolution (configurable trade mode)

Transport must satisfy:

|t_out − t_in| ≤ ε_transport

Where:
	•	ε_transport is explicit and measurable
	•	Determined by codec precision and mode (Timing-Exact or Chunked)

No hidden timing distortion.

3.4 Explicit Time Window Semantics

Each packet carries:

(t_start, t_stop)

Guarantee:

min(E_out) ≥ t_start
max(E_out) ≤ t_stop

If spikes exceed bounds, t_stop must be expanded deterministically.

No implicit truncation.

3.5 Multi-Sender Separation

When multiple neuron IDs are transported:
	•	Sender identity must be preserved
	•	Events must be reconstructable per-sender
	•	No cross-sender ordering corruption

Adapters must support:

Dict[neuron_id → SpikeTrain]

or equivalent deterministic mapping.

4. Graceful Degradation Principle

Under transport noise or precision trade-offs:

Allowed:
	•	Timing precision reduction
	•	Intra-chunk jitter within declared bounds

Not allowed:
	•	Silent spike deletion
	•	Event duplication
	•	Identity corruption
	•	Unbounded timing drift

Transport degradation must be monotonic and measurable.

5. Adapter Compliance Requirements

Any SpikeLink adapter (Neo, Brian2, PyNN, NEST, Nengo, Lava, SpikeInterface, etc.) must satisfy:
	1.	Internal canonical time base: seconds.
	2.	Explicit ms ↔ s conversions documented.
	3.	Two extraction paths when available:
	•	Pre-simulation (generator parameters)
	•	Post-simulation (recorder events)
	4.	Round-trip verification method:
	•	Count invariant
	•	Timing bound verification
	5.	No inference-layer leakage into transport.

Adapters must be transport-pure.

6. Determinism

Given identical input SpikeTrain and codec configuration:

encode → decode → encode

must produce:
	•	Identical spike count
	•	Identical ordering
	•	Timing differences ≤ ε_transport

No stochastic behavior in transport layer.

7. Digital Compatibility

SpikeLink operates inside conventional digital systems.

It does not:
	•	Replace binary computation
	•	Require analog hardware
	•	Depend on neuromorphic silicon

It transports spike symbols without forcing early ADC/DAC collapse or dense tensorization.

8. Application Domains

This contract is relevant when:
	•	Inference is spike/event-native
	•	Timing microstructure carries information
	•	Transport crosses process, node, or hardware boundaries
	•	Latency budgets are bounded and explicit

Examples:
	•	Neuromorphic edge systems
	•	Multi-chip Loihi meshes
	•	Orbital satellite meshes
	•	Robotics event-sensor pipelines
	•	Electrophysiology replay systems


9. Non-Goals

SpikeLink does not guarantee:
	•	Learning convergence
	•	Model accuracy
	•	Biological realism
	•	Synaptic fidelity

It guarantees transport semantics only.

Why This Matters

Most neuromorphic research optimizes:
	•	Energy per inference
	•	On-chip spike efficiency
	•	Plasticity rules

Very few define:

Transport invariants across nodes.

Without a transport contract, distributed spike-native systems collapse into dense digital payloads.

SpikeLink defines the missing layer.