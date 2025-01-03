import ruamel.yaml
import yaml
import json
import os
import copy
import numpy as np
import random
import pandas as pd
from yaml.loader import SafeLoader
import ast
import ruamel
import sys
import logging
import nested_lookup

class Conf:
    """
    Configuration class that describes yaml configurations in class form
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        logging.basicConfig(
            format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d: %H:%M:%S',
            level=logging.DEBUG)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        print(self)
        return self.__dict__ == other.__dict__


def __select_value(value_string, integrate_subconfigs):
    """
    Selects a random value or a value specified by an index from a list of values as specified in the config file using
    the syntax ((value1, value2, value3, ...), index)

    Args:
        value_string: String that contains the values and the index
    Returns:
        Selected value
    """

    values, index = value_string[2:-1].split(")")

    values = values.split(",")
    values = [v.replace(" ", "") for v in values]

    if "random" in index:
        index = np.random.randint(0, len(values))
    else:
        index = index.replace(" ", "").replace(",", "")
        if index.isnumeric():
            index = int(index)
        else:
            logging.warning("Index is not numeric")
            return

    selection = values[index]
    if str(selection).isnumeric():
        selection = float(str(selection))

    if selection[-5:] == ".yaml" and integrate_subconfigs:
        selection = read_yaml(selection, integrate_subconfigs)
    return selection


def __iterate_conf(element, integrate_subconfigs=True):
    """
    Recursive function that iterates through the configuration object and its attributes / sub-objects
    Detects attribute values that represent yaml-paths
    Opens and iterates through sub-yaml-files
    Integrates sub-yaml-files as objects at the placeholder position (path position)

    Args:
        element: Root or sub-node of the yaml file
        integrate_subconfigs: Boolean that specifies if sub-configs referenced by paths should be integrated into the main configuration
    Returns:
        Object that represents the whole yaml-structure with its sub-configs
    """
    attributes = [a for a in dir(element) if a[:2] != "__"]
    for attribute in attributes:
        value = getattr(element, attribute)
        if any(isinstance(value, t) for t in [str, int, bool, float, list, type(None)]):
            if isinstance(value, str):
                if value[-5:] == ".yaml" and integrate_subconfigs:
                    sub_conf = read_yaml(value, integrate_subconfigs)
                    setattr(element, attribute, sub_conf)
                if str(value).isnumeric():
                    value = float(str(value))
                    setattr(element, attribute, value)
                if value[:2] == "((" and value[-1:] == ")":
                    value = __select_value(value, integrate_subconfigs)
                    setattr(element, attribute, value)
        else:
            __iterate_conf(value, integrate_subconfigs)

    return element


def __write_description_file(element, name, filename, comments):
    """
    Recursively iterates through a config object and its attributes
    Writes attributes, values and class dependencies as pyi-file which defines the dynamic object structure and allows
    autocompletion

    Args:
        element: Root or sub-node of the yaml file
        name: Class name of the current node
        filename: Name of the config description file
    """
    attributes = [a for a in dir(element) if a[:2] != "__"]
    for attribute in attributes:
        value = getattr(element, attribute)
        if any(isinstance(value, t) for t in [str, int, bool, float, list, type(None)]):
            pass
        else:
            if not any(isinstance(comments, t) for t in (str, type(None))):
                __write_description_file(value, str(attribute), filename, comments[attribute])
            else:
                __write_description_file(value, str(attribute), filename, comments)
    with open(filename, 'a') as f:
        f.write("class " + name.capitalize() + ":" + "\n")
        for attribute in attributes:
            value = getattr(element, attribute)
            if isinstance(value, (str, int, bool, float, list)):
                if isinstance(value, str):
                    if value.isnumeric():
                        value = float(value)

                f.write("  " + attribute + ": " + str(type(value))[8:-2] + "\n")

                if not isinstance(comments, type(None)):
                    if not isinstance(comments, str):
                        if attribute in comments.ca.items:
                            if not isinstance(comments.ca.items[attribute][2], type(None)):
                                f.write('  "' + str(
                                    comments.ca.items[attribute][2].value[1:].replace('\n', '').lstrip(" ") + '"' + "\n"))
            elif isinstance(value, type(None)):
                f.write("  " + attribute + ": " + "None" + "\n")
            else:
                f.write("  " + attribute + ": " + attribute.capitalize() + "\n")
            delattr(element, attribute)


def __create_class_description(conf, comments):
    """
    Creates a class description by generating a pyi-file
    Calls the recursive writeDescriptionFile function

    Args:
        conf: Configuration object
        comments: Dict file which contains parameters, values and their comments as specified in the config file
    """
    filename = os.path.join(os.path.dirname(__file__), os.path.basename(__file__) + "i")
    with open(filename, 'w') as f:
        f.write("")
    __write_description_file(conf, "Conf", filename, comments)

    # Creates a description of all functions that should be accessible from other python files
    with open(__file__, 'r') as f:
        data = f.read().splitlines()
    for line in data:
        if line[:4] == "def " and line[4:6] != "__":
            with open(filename, 'a') as f:
                f.write(line + "\n")
                f.write("  pass" + "\n")


def read_yaml(path, integrate_subconfigs=True) -> Conf:
    """
    Reads a yaml-config file and creates a config-object
    For a new / adapted yaml-config the createClassInfo attribute should be True in the initial run if
    autocompletion is desired
    Adds comments from the yaml-file to the class description so that they can be viewed within the code by hoovering

    Args:
        path: Path of the configuration yaml file
        integrate_subconfigs: Boolean that specifies if sub-configs referenced by paths should be integrated into the main configuration

    Returns:
        Conf object
    """
    yml = ruamel.yaml.YAML()
    comments = yml.load(open(path))

    conf = json.dumps(comments)
    conf = json.loads(conf, object_hook=lambda d: Conf(**d))
    conf = __iterate_conf(conf, integrate_subconfigs)
    confCopied = copy.deepcopy(conf)
    __create_class_description(confCopied, comments)

    return conf


def conf_from_dict(dict, integrate_subconfigs=True) -> Conf:
    """
    Creates a conf object from a loaded dict

    Args:
        dict: Dictionary that represents the configuration structure
        integrate_subconfigs: Boolean that specifies if sub-configs referenced by paths should be integrated into the main configuration

    Returns:
        Conf object
    """
    conf = json.dumps(dict)
    conf = json.loads(conf, object_hook=lambda d: Conf(**d))
    conf = __iterate_conf(conf, integrate_subconfigs)
    conf_copied = copy.deepcopy(conf)
    if isinstance(dict, ruamel.yaml.comments.CommentedMap):
        __create_class_description(conf_copied, dict)
    else:
        __create_class_description(conf_copied, None)
    return conf


def conf_from_json(file_or_path, integrate_subconfigs=True) -> Conf:
    """
    Creates a conf object from a json-file or a loaded dict
    Args:
        file_or_path: Path to the json-file or loaded dict
        integrate_subconfigs: Boolean that specifies if sub-configs referenced by paths should be integrated into the main configuration
    Returns:
        Conf object
    """
    if os.path.isfile(file_or_path):
        with open(file_or_path, 'r') as f:
            file = json.load(f)
    else:
        file = file_or_path

    return conf_from_dict(file, integrate_subconfigs)


def integrate_sub_confs(path_or_dict) -> dict:
    """
    Reads a yaml-file and integrates sub-configs that are specified by internal paths

    Args:
        path_or_dict: Path of the yaml file that contains the configuration
    Returns:
        Dictionary in which all attributes that describe paths to yaml-sub-configs are
        substituted by the sub-configs contents
    """
    if not isinstance(path_or_dict, dict):
        conf = yaml.full_load(open(path_or_dict))
    else:
        conf = path_or_dict
    conf = json.dumps(conf)
    conf = json.loads(conf, object_hook=lambda d: Conf(**d))
    conf = __iterate_conf(conf)
    yaml.emitter.Emitter.prepare_tag = lambda self, tag: ''
    conf_yaml = yaml.dump(conf)
    conf_dict = yaml.full_load(conf_yaml)
    return conf_dict


def dict_from_conf(conf) -> dict:
    """
    Creates a dictionary from a conf object

    Args:
        conf: Conf object
    Returns:
        Configuration yaml file
    """
    if not isinstance(conf, dict):
        yaml.emitter.Emitter.prepare_tag = lambda self, tag: ''
        conf_yaml = yaml.dump(conf)
        conf_dict = yaml.full_load(conf_yaml)
    else:
        conf_dict = conf
    return conf_dict


def json_from_conf(conf, path):
    """
    Creates a json-file from a conf object

    Args:
        conf: Conf object
        path: Path to the json-file
    """
    conf_dict = dict_from_conf(conf)
    with open(path, 'w') as f:
        json.dump(conf_dict, f, indent=4)


def save_conf(conf, path):
    """
    Saves the given config in yaml-file format into the given path

    Args:
        conf: Configuration object
        path: Save path for the created yaml file
    """
    yaml.emitter.Emitter.prepare_tag = lambda self, tag: ''
    conf_yaml = yaml.dump(conf)
    with open(path, 'w') as f:
        f.write(conf_yaml)


def merge_attributes(conf, merge_section_name) -> Conf:
    """
    Merges attributes which have indices separated by "_" appended to their name
    E.g. attribute_1 = 5, attribute_2 = 6 -> attribute = [5, 6]
    Multiple indices can be used which leads to the creation of multi-dimensional lists as merged values

    Args:
        conf: Conf object
        merge_section_name: Name of the section in the config within which the attributes should be merged
    Returns:
        Conf object with merged attributes
    """

    conf = dict_from_conf(conf)
    merge_section = conf[merge_section_name]

    all_merged = False
    while not all_merged:
        conf_out = {}
        conf_index_attributes = {}
        all_merged = True
        for key in merge_section:
            split_key = key.split("_")
            if len(split_key) > 1:
                if split_key[-1].isnumeric():
                    new_key = "_".join(key.split("_")[:-1])
                    if new_key in conf_index_attributes:
                        conf_index_attributes[new_key].append(merge_section[key])
                    else:
                        conf_index_attributes[new_key] = [merge_section[key]]
                    all_merged = False
                else:
                    conf_out[key] = merge_section[key]
            else:
                conf_out[key] = merge_section[key]

        if len(conf_index_attributes) == 0:
            all_merged = True
        merge_section = {**conf_index_attributes, **conf_out}
    conf[merge_section_name] = merge_section
    conf = conf_from_dict(conf)
    return conf


def conf_to_wandb(hyperparameter_conf, wandb_conf):
    """
    Creates a configuration file suitable for wandb sweeps from a conf object

    Args:
        hyperparameter_conf: Hyperparameter section of the configuration file as conf object
        wandb_conf: Wandb section of the configuration file as conf object
    Returns:
        Dict suitable to create configuration files for wandb sweeps
    """
    # Dictionary that is filled with the configuration
    sweep_conf = {}

    # Creates the general settings within the wandb config
    try:
        early_terminate = {"min_iter": wandb_conf.early_terminate_min_iter,
                           "type": wandb_conf.early_terminate_type}
        sweep_conf["early_terminate"] = early_terminate
    except Exception as e:
        logging.info(e)
        pass
    sweep_conf["method"] = wandb_conf.optimization_method
    metric = {"name": wandb_conf.metric_name,
              "goal": wandb_conf.metric_optimization_goal}
    sweep_conf["metric"] = metric
    sweep_conf["program"] = wandb_conf.program_name

    # Creates the hyperparameter section within the wandb config
    parameters_dict = {}
    hyperparameter_conf = dict_from_conf(hyperparameter_conf)

    for parameter in hyperparameter_conf:
        values = hyperparameter_conf[parameter]

        # Recognizes categorical hyperparameters and adds them to the wandb config
        if str(values).__contains__(";"):
            values = hyperparameter_conf[parameter]
            values = values[0].split(";")
            for i in range(len(values)):
                values[i] = values[i].replace(" ", "")
                if str(values[i]).isnumeric():
                    values = float(str(values[i]))

                if values[i] == "false" or values[i] == "False":
                    values[i] = False
                if values[i] == "true" or values[i] == "True":
                    values[i] = True
            parameters_dict[parameter] = {"values": values}

        # Recognizes uniformly distributed decimal numbers and adds them to the wandb config
        elif str(values[0]).__contains__(".") or str(values[1]).__contains__("."):
            parameters_dict[parameter] = {"distribution": "uniform", "min": values[0], "max": values[1]}

        # Recognizes uniformly distributed integer numbers and adds them to the wandb config
        else:
            parameters_dict[parameter] = {"distribution": "int_uniform", "min": values[0], "max": values[1]}

    # Adds the parameter section to the sweep config and returns the complete wandb config
    sweep_conf["parameters"] = parameters_dict
    return sweep_conf


def create_parameter_combinations(conf_or_dict, search_type, parameter_section_name, number_combinations=None) -> list[Conf]:
    """
    Creates a list of conf objects where each element has a unique hyperparameter value combination determined by the
    specified search type

    Args:
        conf_or_dict: Conf object or dict in which hyperparameter values are specified as intervals
        search_type: Method for selecting the hyperparameter combinations (available: random, grid, one_factor)
        parameter_section_name: Name of the section in the configuration that contains the parameters that should be used for building combinations
        number_combinations: Number of hyperparameter selection that should be selected when using random search

    Returns: List of conf objects with unique hyperparameter sections
    """

    # Creates a dict from the specified dict or conf object
    conf_dict = dict_from_conf(conf_or_dict)

    # Creates copies of the original config and its hyperparameter section
    conf_list = []
    conf_hyperparameters = conf_dict[parameter_section_name]
    original_conf = copy.deepcopy(conf_dict)
    original_hyperparameter_conf = copy.deepcopy(conf_hyperparameters)

    # Random selection of hyperparameter values
    if search_type == "random":
        for i in range(number_combinations):
            conf = copy.deepcopy(original_hyperparameter_conf)
            for parameter in conf:
                # If the current hyperparameter requires value selection
                if isinstance(conf[parameter], list):

                    # If a standard value is defined for the interval, the standard value is removed (only required for one_factor)
                    if isinstance(conf[parameter][0], list):
                        conf[parameter] = conf[parameter][0]

                    # Discrete selection of one value
                    if ";" in str(conf[parameter]):
                        values = str(conf[parameter])[2:-2].split(";")
                        value = random.choice(values)
                        if str(value).__contains__("null"):
                            value = None

                        elif len(str(value)) > 0:
                            if str(value)[0] == " ":
                                value = str(value[1:])

                        if str(value).isnumeric():
                            value = float(value)

                        conf[parameter] = value
                    else:
                        # Selection from interval with defined spacing
                        if len(conf[parameter]) == 3:
                            conf[parameter] = float(random.choice(
                                np.arange(conf[parameter][0], conf[parameter][1] + conf[parameter][2] / 2, conf[parameter][2]).tolist()))

                        # Selection from float interval
                        elif str(conf[parameter][0]).__contains__(".") or str(conf[parameter][1]).__contains__("."):
                            conf[parameter] = random.uniform(conf[parameter][0], conf[parameter][1])

                        # Selection from int interval
                        else:
                            conf[parameter] = float(random.randint(conf[parameter][0], conf[parameter][1]))

            conf_list.append(conf)

    elif search_type == "grid" or search_type == "one_factor":
        conf = copy.deepcopy(original_hyperparameter_conf)
        value_levels = []
        variable_parameters = []
        standard_values = []

        for parameter in conf:
            # If the current hyperparameter requires value selection
            if isinstance(conf[parameter], list):

                # Prints error if no standard value is defined for one_factor method
                if not isinstance(conf[parameter][0], list):
                    if search_type == "one_factor":
                        logging.warning("Standard value must be defined for one-factor-at-a-time method")
                        return []

                # Separates the standard value from the value range and appends it to a list
                else:
                    if search_type == "one_factor":
                        standard_values.append(conf[parameter][1])
                    conf[parameter] = conf[parameter][0]

        for parameter in conf:
            # If the current hyperparameter requires value selection
            if isinstance(conf[parameter], list):
                variable_parameters.append(parameter)

                # Selection from interval with defined spacing
                if len(conf[parameter]) == 3:
                    value_levels.append(np.arange(conf[parameter][0], conf[parameter][1] + conf[parameter][2] / 2,
                                                  conf[parameter][2]).tolist())

                # Discrete selection of one value
                elif ";" in str(conf[parameter]):
                    values = str(conf[parameter])[2:-2].split(";")

                    for i in range(len(values)):
                        if str(values[i]).__contains__("null"):
                            values[i] = None

                        elif len(str(values[i])) > 0:
                            values[i] = values[i].lstrip(" ")

                        if str(values[i]).isnumeric():
                            values[i] = float(str(values[i]))
                    value_levels.append(values)

                # Selection from int interval
                elif "," in str(conf[parameter]) and "." not in str(conf[parameter]):
                    value_levels.append(np.arange(conf[parameter][0], conf[parameter][1] + 0.5).tolist())

                # Float interval not allowed with one_factor method
                else:
                    logging.warning("Float intervals not allowed with grid search and one-factor-at-a-time method")
                    return []
            else:
                if str(conf[parameter]).isnumeric():
                    value = float(conf[parameter])
                else:
                    value = conf[parameter]
                value_levels.append([value])

        # Value creation with grid method
        if search_type == "grid":
            # Creates all possible value combinations from the value levels
            value_combinations = np.array(np.meshgrid(*value_levels))
            if len(value_combinations) > 1:
                value_combinations = value_combinations.T.reshape(-1, len(value_levels))
            else:

                logging.warning("No parameter combinations can be created with the specified configuration")
                return []

            # Overwrites the original hyperparameter config with the new hyperparameters and adds the
            # config to the config list
            for combination in value_combinations:
                conf = copy.deepcopy(original_hyperparameter_conf)
                for i in range(len(variable_parameters)):
                    conf[variable_parameters[i]] = combination[i]

                conf_list.append(conf)

        # Value creation with one_factor method
        else:
            # Overwrites the hyperparameter section of the original config with the new hyperparameters and adds the
            # config to the config list
            for i in range(len(value_levels)):
                for value in value_levels[i]:
                    conf = copy.deepcopy(original_hyperparameter_conf)
                    # Writes the standard value for all hyperparameters
                    for j in range(len(value_levels)):
                        if str(standard_values[j]).isnumeric():
                            standard_values[j] = float(standard_values[j])

                        conf[variable_parameters[j]] = standard_values[j]
                    # Overwrites the value of the currently variable hyperparameters
                        if isinstance(value, list) and str(value[j]).isnumeric():
                            value[j] = float(str(value[j]))
                        conf[variable_parameters[i]] = value

                    conf_list.append(conf)

    # Error when unknown search type is specified
    else:
        logging.warning("Search type not available")
        return []

    # Removes duplicates from the config list
    conf_list = np.array(conf_list)[
        sorted(np.unique([str(conf) for conf in conf_list], return_index=True)[1])].tolist()

    # Overwrites the hyperparameter section of the original config with each hyperparameter selection and transforms
    # each dict config to a conf object
    for i in range(len(conf_list)):
        conf = conf_from_dict(copy.deepcopy(original_conf), False)
        conf.__setattr__(parameter_section_name, conf_from_dict(conf_list[i]))
        conf_list[i] = conf

    return conf_list


def conf_from_wandb_logs(wandb_log_path, original_conf, metric_name=None, optimization_goal=None, number_confs=None) -> list[Conf]:
    """
    Creates a list of config objects where each config has a hyperparameter section extracted from the best wandb runs

    Args:
        wandb_log_path: Path of the logged wandb run data as (csv file)
        original_conf: Conf object of the original config with all parameter sections
        metric_name: Name of the metric used to evaluate wandb runs
        optimization_goal: minimize or maximize
        number_confs: Number of best configs that should be loaded from the run data

    Returns: List of config objects with unique hyperparameter sections extracted from the best wandb runs
    """

    # Creates a dict from the conf object
    original_conf = dict_from_conf(original_conf)
    original_conf["hyperparameters"] = dict_from_conf(merge_attributes(original_conf, "hyperparameters"))

    # Loads the data, removes runs without score and sorts the runs by their scores
    if isinstance(wandb_log_path, str):
        data = pd.read_csv(wandb_log_path)
    else:
        data = wandb_log_path

    if not isinstance(metric_name, type(None)):
        data = data.dropna(subset=[metric_name])
        if optimization_goal == "maximize":
            data = data.sort_values(by=metric_name, ascending=False)
        else:
            data = data.sort_values(by=metric_name, ascending=True)

        # Extracts the number_configs best runs from the data and substitutes nan by None
        data = data.iloc[:number_confs]
    data = data.where(pd.notnull(data), None)

    data_list = []
    for row in data.iterrows():
        row = row[1].to_dict()
        for key in row:
            if row[key] != row[key]:
                row[key] = None
        row = conf_from_dict(row)
        data_list.append(dict_from_conf(row))

    # Creates a list of hyperparameter names contained in the original conf
    parameter_list = list(original_conf["hyperparameters"].keys())

    # Iterates through the selected runs, overwrites the hyperparameter values in a copy of the original config and
    # appends the conf object to a list
    conf_list = []
    for row in data_list:
        conf_dict = copy.deepcopy(original_conf)

        for key in row:
            if isinstance(row[key], str) and row[key].__contains__("["):
                row[key] = ast.literal_eval(row[key])

        hyperparameter_conf_dict = row

        for parameter in hyperparameter_conf_dict:
            if parameter in parameter_list:
                conf_dict["hyperparameters"][parameter] = hyperparameter_conf_dict[parameter]

        conf = conf_from_dict(conf_dict, False)
        conf_list.append(conf)

    return conf_list


def conf_to_csv(save_path, conf, parameter_section_name, metric_names=None, metric_values=None):
    """
    Saves the specified section  of the given config as csv file or adds a row to an existing csv file of config data.
    The remaining config is saved into one column as dict string.

    Args:
        save_path: Path where the csv file should be saved (is also used as load path if the file already exists)
        conf: Conf object or conf_dict
        parameter_section_name: Name of the config section that should be saved parameter-wise in own columns
        metric_names: List of names of the metrics used to evaluate runs
        metric_values: List of values for the metrics of the run with the current config
    """
    if metric_names is None:
        metric_names = []
    if metric_values is None:
        metric_values = []
    conf = dict_from_conf(conf)

    if os.path.exists(save_path):
        data = pd.read_csv(save_path)
    else:
        parameter_list = []
        conf_parameters = conf[parameter_section_name]

        for parameter in conf_parameters:
            parameter_list.append(parameter)
        for metric_name in metric_names:
            parameter_list.append(metric_name)
        parameter_list.insert(0, "pipeline_conf")

        data = pd.DataFrame(columns=parameter_list)

    conf_parameters = conf[parameter_section_name]

    current_data = pd.Series(conf_parameters)

    for metric_name, metric_value in zip(metric_names, metric_values):
        current_data._set_value(metric_name, metric_value)

    conf.pop(parameter_section_name, None)

    current_data._set_value("pipeline_conf", dict_from_conf(conf))

    current_data = pd.DataFrame(current_data).transpose()
    data.columns = current_data.columns
    data = pd.concat([data, current_data], ignore_index=True)

    data.index = np.arange(len(data))

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    data.to_csv(save_path, index=False)


def create_absolute_paths(conf) -> Conf:
    """
    Creates absolute paths by combining the project root path with the relative paths specified in the config
    Args:
        conf: Configuration object
    Returns:
        Configuration object with updated paths
    """
    # Creates absolute paths for the references of external pcb-files
    absolute_path = os.path.dirname(sys.modules['__main__'].__file__)
    for path_name in conf.paths.__dict__:
        if not os.path.isabs(conf.paths.__getattribute__(path_name)):
            path = os.path.join(absolute_path, conf.paths.__getattribute__(path_name))
            conf.paths.__setattr__(path_name, path)
    return conf


def get_values_of_key(conf_or_dict, key) -> list:
    """
    Returns a list of values of a given key in a conf object or dict
    :param conf_or_dict: Config object or dict
    :param key: Key name of the attribute to be searched
    :return: List of values of the given key
    """
    cfg_dict = dict_from_conf(conf_or_dict)

    values = nested_lookup.nested_lookup(key, cfg_dict)
    return values


def get_all_keys(conf_or_dict):
    """
    Returns a list of all keys in a conf object or dict
    :param conf_or_dict: Config object or dict
    :return: List of all keys
    """
    cfg_dict = dict_from_conf(conf_or_dict)
    return nested_lookup.get_all_keys(cfg_dict)