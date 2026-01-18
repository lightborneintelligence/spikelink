"""
NeoAdapter — Integration with the Neo ecosystem.

Provides seamless conversion between SpikeLink SpikeTrain and Neo SpikeTrain
objects, enabling integration with EBRAINS workflows.

Requires: pip install spikelink[neo]
"""

from typing import TYPE_CHECKING, Optional

from spikelink.types.spiketrain import SpikeTrain

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import neo


class NeoAdapter:
    """
    Adapter for Neo SpikeTrain objects.

    Converts between SpikeLink's SpikeTrain and Neo's SpikeTrain,
    preserving metadata and units where possible.

    Example:
        >>> import neo
        >>> import quantities as pq
        >>> from spikelink import NeoAdapter
        >>>
        >>> # From Neo to SpikeLink
        >>> neo_train = neo.SpikeTrain([0.1, 0.2, 0.3] * pq.s, t_stop=1.0 * pq.s)
        >>> our_train = NeoAdapter.from_neo(neo_train)
        >>>
        >>> # From SpikeLink to Neo
        >>> recovered_neo = NeoAdapter.to_neo(our_train)
    """

    @staticmethod
    def from_neo(neo_train: "neo.SpikeTrain") -> SpikeTrain:
        """
        Convert a Neo SpikeTrain to a SpikeLink SpikeTrain.

        Args:
            neo_train: Neo SpikeTrain object

        Returns:
            SpikeLink SpikeTrain
        """
        try:
            import neo
            import quantities as pq
        except ImportError:
            raise ImportError(
                "Neo adapter requires neo and quantities packages. "
                "Install with: pip install spikelink[neo]"
            )

        # Extract times as float array (rescale to seconds)
        times = neo_train.rescale(pq.s).magnitude

        # Extract time bounds
        t_start = float(neo_train.t_start.rescale(pq.s).magnitude)
        t_stop = float(neo_train.t_stop.rescale(pq.s).magnitude)

        # Build metadata from Neo annotations
        metadata = dict(neo_train.annotations) if neo_train.annotations else {}

        # Add Neo-specific info
        if neo_train.name:
            metadata["neo_name"] = neo_train.name
        if neo_train.description:
            metadata["neo_description"] = neo_train.description

        return SpikeTrain(
            times=times,
            t_start=t_start,
            t_stop=t_stop,
            units="s",
            metadata=metadata,
        )

    @staticmethod
    def to_neo(
        train: SpikeTrain,
        units: Optional[str] = None,
    ) -> "neo.SpikeTrain":
        """
        Convert a SpikeLink SpikeTrain to a Neo SpikeTrain.

        Args:
            train: SpikeLink SpikeTrain
            units: Time units for Neo (default: use train.units or 's')

        Returns:
            Neo SpikeTrain object
        """
        try:
            import neo
            import quantities as pq
        except ImportError:
            raise ImportError(
                "Neo adapter requires neo and quantities packages. "
                "Install with: pip install spikelink[neo]"
            )

        # Determine units
        if units is None:
            units = train.units or "s"

        # Get quantities unit
        q_units = getattr(pq, units, pq.s)

        # Create Neo SpikeTrain
        neo_train = neo.SpikeTrain(
            times=train.times * q_units,
            t_start=train.t_start * q_units,
            t_stop=train.t_stop * q_units,
        )

        # Restore metadata as annotations
        if train.metadata:
            # Filter out Neo-specific keys we added
            annotations = {
                k: v for k, v in train.metadata.items() if not k.startswith("neo_")
            }
            neo_train.annotations.update(annotations)

            # Restore name and description if present
            if "neo_name" in train.metadata:
                neo_train.name = train.metadata["neo_name"]
            if "neo_description" in train.metadata:
                neo_train.description = train.metadata["neo_description"]

        return neo_train

    @staticmethod
    def verify_round_trip(
        neo_train: "neo.SpikeTrain",
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Verify that Neo -> SpikeLink -> Neo round trip preserves data.

        Args:
            neo_train: Original Neo SpikeTrain
            tolerance: Maximum allowed error

        Returns:
            True if round trip preserves data within tolerance
        """
        try:
            import numpy as np
            import quantities as pq
        except ImportError:
            raise ImportError("Neo adapter requires neo and quantities packages.")

        # Convert to SpikeLink and back
        our_train = NeoAdapter.from_neo(neo_train)
        recovered = NeoAdapter.to_neo(our_train)

        # Compare times
        original_times = neo_train.rescale(pq.s).magnitude
        recovered_times = recovered.rescale(pq.s).magnitude

        if len(original_times) != len(recovered_times):
            return False

        if len(original_times) == 0:
            return True

        max_error = np.max(np.abs(original_times - recovered_times))
        return max_error <= tolerance
