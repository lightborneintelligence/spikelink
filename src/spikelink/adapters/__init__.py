"""SpikeLink ecosystem adapters."""
def __getattr__(name):
    if name == "NeoAdapter":
        from spikelink.adapters.neo import NeoAdapter
        return NeoAdapter
    if name == "NeoAdapterV2":
        from spikelink.adapters.neo import NeoAdapterV2
        return NeoAdapterV2
    if name == "Brian2AdapterV2":
        from spikelink.adapters.brian2 import Brian2AdapterV2
        return Brian2AdapterV2
    if name == "TonicAdapterV2":
        from spikelink.adapters.tonic import TonicAdapterV2
        return TonicAdapterV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["NeoAdapter", "NeoAdapterV2", "Brian2AdapterV2", "TonicAdapterV2"]
