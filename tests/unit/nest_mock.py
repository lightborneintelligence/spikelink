"""
nest_mock.py — Faithful PyNEST API mock for SpikeLink CI testing
==================================================================

Replicates the subset of the NEST simulator Python interface
needed by the SpikeLink NestAdapter:

    nest.ResetKernel()
    nest.Create(model, n, params)   → NodeCollection
    nest.Connect(pre, post, ...)
    nest.Simulate(t)
    nest.GetStatus(nodes, keys)
    nest.SetStatus(nodes, params)
    node.get(key)
    node.set(**params)

Models supported:
    spike_generator   — inject spike times (ms)
    spike_recorder    — record spikes (events: times + senders)
    iaf_psc_alpha     — integrate-and-fire neuron (simplified)
    parrot_neuron     — repeats incoming spikes 1:1
    poisson_generator — rate-based spike source (simplified)

All times are in milliseconds, matching real NEST convention.

Lightborne Intelligence · Dallas TX
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union


# ── Global kernel state ──────────────────────────────────────

_nodes: Dict[int, dict] = {}
_connections: List[dict] = []
_next_id: int = 1
_resolution: float = 0.1  # ms
_simulated_time: float = 0.0


def ResetKernel():
    """Reset NEST kernel to initial state."""
    global _nodes, _connections, _next_id, _resolution, _simulated_time
    _nodes = {}
    _connections = []
    _next_id = 1
    _resolution = 0.1
    _simulated_time = 0.0


def GetKernelStatus(key: str = None):
    """Get kernel status."""
    status = {
        "resolution": _resolution,
        "biological_time": _simulated_time,
        "network_size": len(_nodes),
    }
    if key:
        return status.get(key)
    return status


def SetKernelStatus(params: dict):
    """Set kernel status."""
    global _resolution
    if "resolution" in params:
        _resolution = params["resolution"]


# ── NodeCollection ───────────────────────────────────────────

class NodeCollection:
    """Mock of NEST's NodeCollection (tuple of node GIDs)."""

    def __init__(self, gids: List[int]):
        self._gids = list(gids)

    def __len__(self):
        return len(self._gids)

    def __iter__(self):
        for gid in self._gids:
            yield NodeCollection([gid])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return NodeCollection(self._gids[key])
        if isinstance(key, int):
            if key < 0:
                key = len(self._gids) + key
            return NodeCollection([self._gids[key]])
        raise TypeError(f"Invalid index type: {type(key)}")

    def __repr__(self):
        return f"NodeCollection({self._gids})"

    def __eq__(self, other):
        if isinstance(other, NodeCollection):
            return self._gids == other._gids
        return False

    def __add__(self, other):
        if isinstance(other, NodeCollection):
            return NodeCollection(self._gids + other._gids)
        raise TypeError(f"Cannot add NodeCollection and {type(other)}")

    @property
    def global_id(self):
        """Return first GID (for single-node collections)."""
        if len(self._gids) == 1:
            return self._gids[0]
        return self._gids

    @property
    def tolist(self):
        return list(self._gids)

    def get(self, key: Union[str, List[str]] = None, **kwargs) -> Any:
        """Get parameter(s) from node(s) — matches NEST NodeCollection.get()."""
        if len(self._gids) == 0:
            return {} if key is None else None

        if len(self._gids) == 1:
            node = _nodes[self._gids[0]]
            if key is None:
                return dict(node)
            if isinstance(key, str):
                return node.get(key)
            if isinstance(key, list):
                return {k: node.get(k) for k in key}
        else:
            # Multiple nodes
            if key is None:
                return [dict(_nodes[gid]) for gid in self._gids]
            if isinstance(key, str):
                return [_nodes[gid].get(key) for gid in self._gids]
            if isinstance(key, list):
                return [{k: _nodes[gid].get(k) for k in key} for gid in self._gids]

        return None

    def set(self, **params):
        """Set parameter(s) on node(s)."""
        for gid in self._gids:
            if gid in _nodes:
                _nodes[gid].update(params)


# ── Create / Connect / Simulate ──────────────────────────────

def Create(model: str, n: int = 1, params: dict = None) -> NodeCollection:
    """Create NEST nodes."""
    global _next_id

    gids = []
    for _ in range(n):
        gid = _next_id
        _next_id += 1
        gids.append(gid)

        # Initialize node based on model
        node = {
            "model": model,
            "global_id": gid,
        }

        if model == "spike_generator":
            node["spike_times"] = []
            node["origin"] = 0.0
            node["start"] = 0.0
            node["stop"] = float("inf")

        elif model == "spike_recorder":
            node["events"] = {"times": [], "senders": []}
            node["n_events"] = 0
            node["origin"] = 0.0
            node["start"] = 0.0
            node["stop"] = float("inf")

        elif model == "iaf_psc_alpha":
            node["V_m"] = -70.0
            node["V_th"] = -55.0
            node["V_reset"] = -70.0
            node["tau_m"] = 10.0
            node["C_m"] = 250.0
            node["t_ref"] = 2.0

        elif model == "parrot_neuron":
            pass  # Parrot just repeats spikes

        elif model == "poisson_generator":
            node["rate"] = 0.0
            node["origin"] = 0.0
            node["start"] = 0.0
            node["stop"] = float("inf")

        else:
            # Generic node
            pass

        # Apply user params
        if params:
            node.update(params)

        _nodes[gid] = node

    return NodeCollection(gids)


def Connect(
    pre: NodeCollection,
    post: NodeCollection,
    conn_spec: Union[str, dict] = "all_to_all",
    syn_spec: dict = None,
):
    """Connect nodes."""
    if syn_spec is None:
        syn_spec = {}

    delay = syn_spec.get("delay", 1.0)  # Default 1 ms delay
    weight = syn_spec.get("weight", 1.0)

    for pre_gid in pre._gids:
        for post_gid in post._gids:
            _connections.append({
                "source": pre_gid,
                "target": post_gid,
                "delay": delay,
                "weight": weight,
            })


def Simulate(t: float):
    """Simulate the network for t milliseconds."""
    global _simulated_time

    end_time = _simulated_time + t

    # Collect all spike events
    spike_events = []  # (time, source_gid, target_gid)

    # Process spike generators
    for gid, node in _nodes.items():
        if node["model"] == "spike_generator":
            spike_times = node.get("spike_times", [])
            for st in spike_times:
                if _simulated_time <= st < end_time:
                    # Find connections from this generator
                    for conn in _connections:
                        if conn["source"] == gid:
                            arrival_time = st + conn["delay"]
                            if arrival_time < end_time:
                                spike_events.append((arrival_time, gid, conn["target"]))

        elif node["model"] == "poisson_generator":
            rate = node.get("rate", 0.0)
            if rate > 0:
                # Generate Poisson spikes
                n_spikes = np.random.poisson(rate * t / 1000.0)
                spike_times = np.sort(np.random.uniform(_simulated_time, end_time, n_spikes))
                for st in spike_times:
                    for conn in _connections:
                        if conn["source"] == gid:
                            arrival_time = st + conn["delay"]
                            if arrival_time < end_time:
                                spike_events.append((arrival_time, gid, conn["target"]))

    # Sort events by time
    spike_events.sort(key=lambda x: x[0])

    # Deliver spikes to recorders
    for arrival_time, source_gid, target_gid in spike_events:
        target = _nodes.get(target_gid)
        if target is None:
            continue

        if target["model"] == "spike_recorder":
            target["events"]["times"].append(arrival_time)
            target["events"]["senders"].append(source_gid)
            target["n_events"] = len(target["events"]["times"])

        elif target["model"] == "parrot_neuron":
            # Parrot repeats to its targets
            for conn in _connections:
                if conn["source"] == target_gid:
                    downstream_target = _nodes.get(conn["target"])
                    if downstream_target and downstream_target["model"] == "spike_recorder":
                        new_arrival = arrival_time + conn["delay"]
                        if new_arrival < end_time:
                            downstream_target["events"]["times"].append(new_arrival)
                            downstream_target["events"]["senders"].append(target_gid)
                            downstream_target["n_events"] = len(downstream_target["events"]["times"])

    _simulated_time = end_time


# ── GetStatus / SetStatus ────────────────────────────────────

def GetStatus(nodes: NodeCollection, keys: Union[str, List[str]] = None):
    """Get status of nodes."""
    if len(nodes._gids) == 1:
        node = _nodes.get(nodes._gids[0], {})
        if keys is None:
            return node
        if isinstance(keys, str):
            return node.get(keys)
        if isinstance(keys, list):
            return {k: node.get(k) for k in keys}
    else:
        result = []
        for gid in nodes._gids:
            node = _nodes.get(gid, {})
            if keys is None:
                result.append(node)
            elif isinstance(keys, str):
                result.append(node.get(keys))
            elif isinstance(keys, list):
                result.append({k: node.get(k) for k in keys})
        return result


def SetStatus(nodes: NodeCollection, params: Union[dict, List[dict]]):
    """Set status of nodes."""
    if isinstance(params, dict):
        for gid in nodes._gids:
            if gid in _nodes:
                _nodes[gid].update(params)
    elif isinstance(params, list):
        for gid, p in zip(nodes._gids, params):
            if gid in _nodes:
                _nodes[gid].update(p)


# ── Utility functions ────────────────────────────────────────

def GetConnections(source: NodeCollection = None, target: NodeCollection = None):
    """Get connections matching source/target criteria."""
    result = []
    for conn in _connections:
        if source is not None and conn["source"] not in source._gids:
            continue
        if target is not None and conn["target"] not in target._gids:
            continue
        result.append(conn)
    return result


def PrintNetwork():
    """Print network summary."""
    print(f"Network: {len(_nodes)} nodes, {len(_connections)} connections")
    for gid, node in _nodes.items():
        print(f"  [{gid}] {node['model']}")


# ── Module-level attributes for NEST compatibility ──────────

resolution = _resolution


def __getattr__(name):
    if name == "resolution":
        return _resolution
    raise AttributeError(f"module 'nest_mock' has no attribute '{name}'")
