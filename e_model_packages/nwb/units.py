"""NEURON units to NWB units conversion."""


class UnitConverter:
    """Converts arrays or single values."""

    def __init__(self, conversion_rate):
        self.conversion_rate = conversion_rate

    def convert_array(self, arr):
        """Converts the unit of the input array."""
        return arr * self.conversion_rate

    def convert_value(self, val):
        """Converts the unit of the input value."""
        return val * self.conversion_rate
