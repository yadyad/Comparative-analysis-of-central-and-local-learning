from datetime import datetime

from data_preprocessing import DataPreprocessing
from pytorch_image_classification.centralised_learning import CentralLearning
from pytorch_image_classification.federated_learning.FedNova.fednovaservernew import FedNovaServerNew
from pytorch_image_classification.federated_learning.FedProxServer import FedProxServer
from pytorch_image_classification.federated_learning.FederatedAveragingServer import FederatedAveragingServer
from pytorch_image_classification.federated_learning.scaffold.serverscaffold import SCAFFOLD
from pytorch_image_classification.local_learning_pytorch import LearningLocal

start_time = datetime.now()  # Format the date and time as a string
date_time_str = start_time.strftime("%Y-%m-%d%H-%M-%S")
cumulative_log_path = []
for i in range(1):
    log_paths_per_iteration = []

    D = DataPreprocessing()
    local = LearningLocal(f'{date_time_str}__it-{i}__local')
    local.train_model(local.train_parts, local.val_parts, local.test_parts)
    log_paths_per_iteration.append(local.log.logging_path)

    central = CentralLearning(f'{date_time_str}__it-{i}__central')
    central.train_central_basic(central.train, central.val, central.test)
    log_paths_per_iteration.append(central.log.logging_path)

    fed_avg = FederatedAveragingServer(f'{date_time_str}__it-{i}__fed')
    log_paths_per_iteration.append(fed_avg.log.logging_path)

    fed_prox = FedProxServer(f'{date_time_str}__it-{i}__fed')
    log_paths_per_iteration.append(fed_avg.log.logging_path)

    scaffold = SCAFFOLD(f'{date_time_str}__it-{i}__fed')
    log_paths_per_iteration.append(scaffold.log.logging_path)

    fed_nova = FedNovaServerNew(f'{date_time_str}__it-{i}__fed')
    log_paths_per_iteration.append(scaffold.log.logging_path)
    cumulative_log_path.append(log_paths_per_iteration)
end_time = datetime.now()

print(f" central start time: {start_time} Runtime: {end_time - start_time}")
print(f"log_paths: {cumulative_log_path}")