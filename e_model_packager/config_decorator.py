"""Contains classes to be used in parsing Luigi configs."""

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


class ConfigDecorator:
    """Decorator class over LuigiConfigParser.

    Attributes:
        luigi_config: Luigi config parser object.
    """

    def __init__(self, luigi_config):
        """Constructor."""
        self.luigi_config = luigi_config

    def get(self, section, option, delim=","):
        """The get method that treats lists.

        Args:
            section: a section of config
            option: an option of the section
            delim: delimeter to be used in case of a list
        """
        default_val = self.luigi_config.get(section, option, None)

        if isinstance(default_val, str) and delim in default_val:
            default_val = default_val.split(delim)

        return default_val
