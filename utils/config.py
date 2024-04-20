class _C2G_CONFIG:
    """
    This class provides functionalities for managing configurations and modules within the C2G system.

    Methods
    - clean_config(): Resets the configuration dictionary.
    - clean_dir(): Clears the module directory and module list.
    - display_available_modules(): Prints the names of available modules.
    - update_inverse_list(): Updates the inverse dictionary of module names.

    Attributes:
    - _C: A dictionary storing configuration settings.
    - _dir: A dictionary mapping module names to their corresponding objects.
    - _module_list: A list containing names of registered modules.

    Example:
        ```python
        >>> @register_module('add')
        >>> def add(a, b):
        >>>     return a+b
        >>> add(-10, 12)
        2
        >>> cfg.display_available_modules()
        Available Modules:
        ADD
        >>> cfg.clean_dir()
        >>> cfg.display_available_modules()
        Available Modules:
        ```
    """
    _C = {}
    _dir = {}
    _module_list = []

    def clean_config(self):
        self._C = {}

    def clean_dir(self):
        self._dir = {}
        self._module_list = []

    def display_available_modules(self):
        print('Available Modules:')
        for module_name, _ in self._dir.items():
            print(module_name)

    def update_inverse_list(self):
        self._inv_dir = {value : key for key, value in self._dir.items()}


class register_module:
    """
    This class is a decorator used for registering modules within the C2G system.

    Methods
    - __init__(self, name): Initializes the module with a given name.
    - __call__(self, module): Registers the module with the provided name and updates the module list.

    Examples
        ```python
        >>> @register_module('add')
        >>> def add(a, b):
        >>>     return a+b
        >>> add(-10, 12)
        2
        >>> cfg.display_available_modules()
        Available Modules:
        ADD
        ```
    """
    def __init__(self, name):

        self.name = name.upper()
    def __call__(self, module):
        cfg._dir[self.name] = module
        cfg.update_inverse_list()
        cfg._module_list.append(self.name)
        return cfg._dir[self.name]
