from utils.utils import _C
# from utils.model_configs import DEFAULT_CONFIGS


class C2GModelBase:
    """
    Base class for managing configurations and parameters of a Corpus2GPT model.

    Attributes:
        DEFAULT_CONFIGS (dict): A dictionary containing default configurations
        for the model.
        BASE_LABEL (str): Base label used to identify the model instance.

    Methods:
        __init__(DEFAULT_CONFIGS, BASE_LABEL="_C2GModelName"):
            Initializes the C2GModelBase with default configurations and a
            base label.

        restore_default_model(cfg):
            Restores the default model configurations.

        delete_model():
            Deletes the model configurations.

        change_param(key, value):
            Changes the value of a specific parameter.

        load_from_config(config):
            Loads configurations from a given dictionary and applies them to
            the model instance.
    """

    def __init__(self, DEFAULT_CONFIGS, BASE_LABEL="_C2GModelBase"):
        """
        Initializes the C2GModelBase instance.

        Args:
            DEFAULT_CONFIGS (dict): A dictionary containing default configurations
            for the model.
            BASE_LABEL (str, optional): Base label used to identify the model
            instance. Default is "_C2GModelBase".
        """
        self.BASE_LABEL = BASE_LABEL
        self.DEFAULT_CONFIGS = DEFAULT_CONFIGS
        self._C = _C
        self.restore_default_model(DEFAULT_CONFIGS)

    def restore_default_model(self, cfg):
        """
        Restores the default model configurations.

        Args:
            cfg (dict): A dictionary containing default configurations for the model.
        """
        _C.add_dir(**{self.BASE_LABEL: cfg})

    def delete_model(self):
        """Deletes the model configurations."""
        _C.dir[self.BASE_LABEL] = {}
        self.__init__(self.DEFAULT_CONFIGS, self.BASE_LABEL)

    def change_param(self, key, value):
        """
        Changes the value of a specific parameter.

        Args:
            key (str): The name of the parameter.
            value: The new value for the parameter.

        Raises:
            KeyError: If the parameter name doesn't exist in the dictionary.
        """
        dictionary = _C.get_dir()
        if key in dictionary[self.BASE_LABEL]:
            dictionary[self.BASE_LABEL][key] = value
        else:
            raise KeyError(f"The parameter name '{key}' doesn't exist in the dictionary.")
        _C.clean_dir()
        _C.add_dir(**dictionary)

    def load_from_config(self, config):
        """
        Loads configurations from a given dictionary and applies them to the
        model instance.

        Args:
            config (dict): A dictionary containing configurations to be loaded.
        """
        self.delete_model()
        _C.dir[str(self.BASE_LABEL)].update(**config)