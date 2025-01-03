class Binary:
  activation: str
  loss: str
class Central:
  epoch: int
class Class_names_dict:
  0: str
  1: str
  10: str
  11: str
  12: str
  13: str
  14: str
  15: str
  2: str
  3: str
  4: str
  5: str
  6: str
  7: str
  8: str
  9: str
class Class_number_dict:
  Background Image: int
  Pin fehlt: int
  Pseudofehler: int
  THT Bauteil Montage falsch: int
  THT Bauteil falsch: int
  THT Bauteil fehlt: int
  THT Bauteil liegt nicht auf: int
  THT Bauteil mechan beschaedigt: int
  THT Bauteil verpolt: int
  THT Bauteil versetzt: int
  THT Loetbruecke: int
  THT Loetdurchstieg mangelhaft: int
  THT Loetst.Form mangelhaft: int
  THT Loetstelle offen: int
  THT Lotkugeln: int
  good: int
class Crop_center:
  on: bool
  size: int
class Flipping:
  on: bool
  probability: float
class Mirroring:
  on: bool
  probability: float
class Normalize:
  mean: list
  on: bool
  standard_deviation: list
class Resize:
  on: bool
  size: int
class Rotation:
  max_degrees: int
  min_degrees: int
  on: bool
class Data_augmentation:
  augment: bool
  crop_center: Crop_center
  flipping: Flipping
  mirroring: Mirroring
  normalize: Normalize
  resize: Resize
  rotation: Rotation
class Federated_learning:
  batch_size: int
  batch_size_val: int
  communication_rounds: int
  global_lr: float
  gmf: float
  is_same_initial: bool
  learning_rate: float
  learning_rate_decay_gamma: bool
  local_epochs: int
  random_seed: int
  sample: int
  seed: int
  test_ratio_per_client: float
  validation_size: float
  var_local_epochs: bool
  var_max_epochs: int
  var_min_epochs: int
class Image:
  base_path: list
  column_drop: list
  column_selected: list
  filter_rows: list
  image_feature_name: str
  remove_rows: list
  size_x: int
  size_y: int
class Local:
  epoch: int
class Vgg:
  epochs: int
  learning_rate: float
class Conv_1:
  activation: str
  kernel_size: int
  no_of_filter: int
  strides: int
class Conv_2:
  activation: str
  kernel_size: int
  no_of_filter: int
  strides: int
class Conv_3:
  activation: str
  kernel_size: int
  no_of_filter: int
  strides: int
class Dense_1:
  activation: str
  in_features: int
  out_features: int
class Dense_2:
  activation: str
  in_features: int
  out_features: int
class Dense_3:
  activation: str
  in_features: int
class Pool:
  kernel_size: int
  strides: int
class Basic_convolution:
  conv_1: Conv_1
  conv_2: Conv_2
  conv_3: Conv_3
  dense_1: Dense_1
  dense_2: Dense_2
  dense_3: Dense_3
  dropout: float
  epochs: int
  in_channel: int
  learning_rate: float
  loss: str
  metrics: str
  pool: Pool
class Conv_1:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Conv_2:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Conv_3:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Conv_4:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Conv_5:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Conv_6:
  activation: str
  kernel_size: int
  no_of_filter: int
  padding: str
  strides: int
class Dense_1:
  activation: str
  in_features: int
  no_of_neurons: int
  out_features: int
class Dense_2:
  activation: str
  in_features: int
  no_of_neurons: int
class Dropout_1:
  percent: float
class Dropout_2:
  percent: float
class Pool_1:
  kernel_size: int
  strides: int
class Pool_2:
  kernel_size: int
  strides: int
class Pool_3:
  kernel_size: int
class Improved_convolution:
  conv_1: Conv_1
  conv_2: Conv_2
  conv_3: Conv_3
  conv_4: Conv_4
  conv_5: Conv_5
  conv_6: Conv_6
  dense_1: Dense_1
  dense_2: Dense_2
  dropout_1: Dropout_1
  dropout_2: Dropout_2
  epochs: int
  in_channel: int
  learning_rate: float
  loss: str
  metrics: str
  pool_1: Pool_1
  pool_2: Pool_2
  pool_3: Pool_3
class Base_model:
  include_top: bool
  trainable: bool
  weights: str
class Dense_1:
  activation: str
  no_of_neurons: int
class Dense_2:
  activation: str
  no_of_neurons: int
class Dense_3:
  activation: str
  no_of_neurons: int
class Dense_4:
  activation: str
  no_of_neurons: int
class Resnet_50:
  base_model: Base_model
  dense_1: Dense_1
  dense_2: Dense_2
  dense_3: Dense_3
  dense_4: Dense_4
  epochs: int
  learning_rate: float
  loss: str
  metrics: str
class Local_learning:
  VGG: Vgg
  basic_convolution: Basic_convolution
  feature_based_sampling: bool
  fraction: float
  improved_convolution: Improved_convolution
  random_sampling: bool
  random_state: int
  resnet_50: Resnet_50
  test_size: float
  validation_size: float
class Metrics:
  beta: int
  min_threshold: float
  name: str
  threshold: float
class Multi_class:
  activation: str
  loss: str
  ""
class Paths:
  logging_path: str
class Plot:
  plot_stat_data_loader: bool
  "#constants"
class Splitting:
  min_samples_per_split: int
  random_state: int
  split_type: str
  splitting_feature_name: str
  train_val_test_split_ratio: float
class Conf:
  Number_of_classes: int
  background_image_class_name: str
  batch_size: int
  binary: Binary
  central: Central
  class_names_dict: Class_names_dict
  class_number_dict: Class_number_dict
  classification_type: str
  "binary, multiclass"
  data_augmentation: Data_augmentation
  dataset_feature_name: str
  device: str
  "cuda or cpu"
  federated_learning: Federated_learning
  fetch_image: str
  image: Image
  image_feature_name: str
  image_index_feature: str
  local: Local
  local_learning: Local_learning
  logging: bool
  ""
  lr: float
  metrics: Metrics
  model: str
  mu: float
  multi_class: Multi_class
  no_of_classes: int
  no_of_local_machines: int
  optimizer: str
  ""
  paths: Paths
  plot: Plot
  positive_classes: str
  random_oversampling: bool
  rho: float
  sampling_type: str
  sampling_type_weighted: str
  sd_iter: int
  ""
  separation_type: str
  "split data for each client ::: random, feature"
  shuffle_train: bool
  shuffle_val: bool
  splitting: Splitting
  target_feature_name: str
  target_feature_number: str
  ""
  use_standard_dataset: bool
  weight_decay: float
  weighted_sampling: bool
def read_yaml(path, integrate_subconfigs=True) -> Conf:
  pass
def conf_from_dict(dict, integrate_subconfigs=True) -> Conf:
  pass
def conf_from_json(file_or_path, integrate_subconfigs=True) -> Conf:
  pass
def integrate_sub_confs(path_or_dict) -> dict:
  pass
def dict_from_conf(conf) -> dict:
  pass
def json_from_conf(conf, path):
  pass
def save_conf(conf, path):
  pass
def merge_attributes(conf, merge_section_name) -> Conf:
  pass
def conf_to_wandb(hyperparameter_conf, wandb_conf):
  pass
def create_parameter_combinations(conf_or_dict, search_type, parameter_section_name, number_combinations=None) -> list[Conf]:
  pass
def conf_from_wandb_logs(wandb_log_path, original_conf, metric_name=None, optimization_goal=None, number_confs=None) -> list[Conf]:
  pass
def conf_to_csv(save_path, conf, parameter_section_name, metric_names=None, metric_values=None):
  pass
def create_absolute_paths(conf) -> Conf:
  pass
def get_values_of_key(conf_or_dict, key) -> list:
  pass
def get_all_keys(conf_or_dict):
  pass
