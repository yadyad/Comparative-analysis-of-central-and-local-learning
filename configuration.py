from config_parser import config_parser


class Configuration:
    """
        class configuration used to parse config file and return parameters
    """
    _instance = None

    def __new__(cls, file_path='C:/Users/yadhu/Desktop/thesis_code/image-classification-master/configuration.yaml'):
        #override __new__ function to open and parse configuration.yaml whenever function is started
        if cls._instance is None:
            with open(file_path, 'r') as file:
                cls._instance = config_parser.read_yaml(file_path)
        return cls._instance
