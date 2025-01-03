import json
import os
from types import MappingProxyType

import numpy as np
from datetime import datetime
from config_parser import config_parser
import sys
from collections.abc import ValuesView, KeysView, ItemsView


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def logging_setup(logging_path, logging):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    filename = f'{logging_path}/{logging_path.split("/")[-1]}_logs'
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.info(filename)


def get_sampling(cfg):
    if cfg.random_oversampling:
        return "_r_"
    elif cfg.weighted_sampling:
        return "_w_"
    else:
        return "_n_"


class ResultsLogger:
    def __init__(self, conf_or_dict: [config_parser.Conf, dict], classname=None, id=None):
        """
        Initialize the ResultsLogger object.
        :param classname: the name of the class from which results_logger is called.
        :param id: identifier for iteration for calculating mean and standard deviation results.
        :param conf_or_dict: Configuration object or dictionary with configuration.
        """
        cfg = conf_or_dict
        self.logging_path = cfg.paths.logging_path
        cfg_dict = config_parser.dict_from_conf(cfg)

        date_time = datetime.now()
        # date_time = date_time.strftime('%Y_%m_%d_%H_%M_%S')
        date_time = date_time.strftime('%Y_%m_%d_%H_%M')
        self.logging_output = {"_s": get_sampling(cfg), "_sep": cfg.separation_type, "start_time": date_time,
                               "main_file": sys.argv[0],
                               "config_file": cfg_dict}
        self.filename = (
            f"_s{self.logging_output['_s']}00_sep{self.logging_output['_sep']}{self.logging_output['start_time']}"
            #f"_{os.path.basename(self.logging_output['main_file']).split('.')[0]}")
            f"_{classname}")
        if id:
            self.logging_path = self.logging_path + f'{id}/' + self.filename
        else:
            self.logging_path = self.logging_path + self.filename
        os.makedirs(self.logging_path, exist_ok=True)
        print(self.logging_output)

    def save_log(self):
        """Save the log to a json file."""

        # Ensure all parts of logging_output are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, ValuesView):  # Handle dict_values
                return list(obj)
            elif isinstance(obj, KeysView):  # Handle dict_keys
                return list(obj)
            elif isinstance(obj, ItemsView):  # Handle dict_items
                return list(obj)
            elif isinstance(obj, MappingProxyType):  # Handle mappingproxy
                return dict(obj)
            elif isinstance(obj, np.ndarray):  # Handle numpy arrays
                return obj.tolist()
            # Add other types that need conversion if necessary
            return obj

        # Recursively make the logging output serializable
        serializable_output = make_serializable(self.logging_output)

        # Ensure the logging path exists
        os.makedirs(self.logging_path, exist_ok=True)

        # Construct the file name
        file_name = f"_s{self.logging_output['_s']}00_sep{self.logging_output['_sep']}{serializable_output['start_time']}_{os.path.basename(serializable_output['main_file']).split('.')[0]}.json"

        # Save to JSON file
        with open(os.path.join(self.logging_path, file_name), "w") as file:
            json.dump(serializable_output, file, indent=3, cls=NumpyEncoder)

    def add_to_log(self, path, data_dict):
        """
        Add data to the log at the given path.
        :param path: Path of keys within the log dictionary where the data should be inserted.
        :param data_dict: Dictionary with data to be inserted.
        """
        if not path:  # Check if path is empty
            self.logging_output.update(data_dict)
        else:
            def insert_dict(d, keys, value):
                """Recursive function to insert value at given keys path in dictionary d."""
                current_key = keys[0]
                if len(keys) == 1:
                    if current_key in d:
                        d[current_key].update(value)
                    else:
                        d[current_key] = value
                else:
                    if current_key not in d:
                        d[current_key] = {}
                    insert_dict(d[current_key], keys[1:], value)

            insert_dict(self.logging_output, path, data_dict)
