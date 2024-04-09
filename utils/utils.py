from .config import *

class BASE_C2G:
    """
    This is the base class for C2G models.

    Class Methods
        serialize_c2g(cls, config): Serializes the configuration into a dictionary.
        deserialize_c2g(self, config): Deserializes the configuration from a dictionary.
        construct_model(self, configs): Abstract method for constructing a model. This needs to be implemented by subclasses.

    Examples
        ```
        >>> @register_module('test_module')
        >>> class test_module(BASE_C2G):
        >>>     def __init__(self, a, b):
        >>>         self._config = {'a':a, 'b':b}
        >>>         self.a = a
        >>>         self.b = b
        >>>     def __call__(self):
        >>>         return self.a+self.b

        >>> sub_module = test_module(1, 2)
        >>> sub_module()
        3

        >>> @register_module('master_module')
        >>> class master_module(BASE_C2G):
        >>>     def __init__(self, sub_module):
        >>>         self._config = {'sub_module': sub_module}
        >>>         self.sub_module = sub_module

        >>>     def __call__(self):
        >>>         print(self.sub_module())

        >>> master_mod = master_module(sub_module)
        >>> master_mod()
        3

        >>> save_master_module = BASE_C2G()
        >>> save_master_module.serialize_c2g(master_mod._config)
        {'BASE_C2G': {'SUB_MODULE': {'TEST_MODULE': {'a': 1, 'b': 2}}}}
        >>> cfg._module_list
        ['TEST_MODULE', 'MASTER_MODULE']
        ```
    """
    @classmethod
    def serialize_c2g(cls, config):
        cls_config = {}
        for keyword, value in config.items():
            if value.__class__.__name__.upper() not in cfg._module_list:
            #if value.__name__.upper() not in cfg._module_list:
                cls_config[keyword] = value
            else:
                cls_config[keyword.upper()] = value.serialize_c2g(value._config)
        cfg._C = {cls.__name__.upper(): cls_config}
        return cfg._C

    def has_sub_dicts(self, config):
        for key, val in config.items():
            if isinstance(val, dict):
                return False
        return True

    def deserialize_c2g(self, config):
        for model, params in config.items():
            pass
        return cfg._dir[model](**params)

    def construct_model(self, configs):
        raise NotImplementedError('`construct_model` method is not implemented. To implement it, define it on the subclasses of BASE_C2G Models.')
