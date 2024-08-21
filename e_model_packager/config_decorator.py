"""Contains classes to be used in parsing Luigi configs."""


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
        default_val = self.luigi_config.get(section, option)

        if isinstance(default_val, str) and delim in default_val:
            default_val = default_val.split(delim)

        return default_val
