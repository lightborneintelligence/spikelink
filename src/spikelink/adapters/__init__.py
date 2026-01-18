"""SpikeLink ecosystem adapters."""

# Lazy imports to avoid requiring optional dependencies
def __getattr__(name):
    if name == "NeoAdapter":
        from spikelink.adapters.neo import NeoAdapter
        return NeoAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["NeoAdapter"]
