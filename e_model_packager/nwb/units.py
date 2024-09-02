"""NEURON units to NWB units conversion."""

"""
Copyright 2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


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
