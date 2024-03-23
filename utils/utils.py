
class C2G_config:
    """ Configuration class for C2G GPT Model.
    """
    
    VERSION = "1.0.0"
    dir = {}

    def __init__(self):
        """ Checks for the configuration of different modules
        such as version, dir, attributes, etc.
        """

        assert hasattr(self, 'VERSION'), "Error! C2G files seem incomplete or corrupt!"
        assert hasattr(self, 'dir'), "Error! The dir attribute hasn't been initialized."
    
    def get_version(self):
        """ Displays the version of the build.
        `>>> print(c2g_cfg.get_version())`
        `1.0.0`
        """
        print(str(self.VERSION))
    
    def get_dir(self):
        """ Returns the Configuration of the project loaded into the memory
        `>>> print(c2g_cfg.get_dir())`
        `{'component_1' : 1, 'parameter_2_val' : 0.001}`
        """
        return self.dir
    
    def add_dir(self, **kwargs):
        """ Add configuration to dir.
        `>>> c2g_cfg.add_dir(settings = MySettings)`
        """
        self.dir.update(kwargs)
    
    def pop_dir(self, key):
        """ Remove certain configurations from dir.
        >>> c2g_cfg.pop_dir(parameter_2_val)
        """
        self.dir.pop(key, None)
    
    def clean_dir(self):
        """ Close the session without saving.
        **Warning**: This will permanently erase the whole session!

        `c2g_cfg.clean_dir()`
        """
        self.dir = {}

  _C = C2G_config()
