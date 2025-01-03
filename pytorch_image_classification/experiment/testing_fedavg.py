from datetime import datetime

from configuration import Configuration
from pytorch_image_classification.federated_learning.FederatedAveragingServer import FederatedAveragingServer
cfg = Configuration()
start_time = datetime.now()  # Format the date and time as a string
date_time_str = start_time.strftime("%Y-%m-%d%H-%M-%S")
log_paths = []
"""
    code for running experiment on federated averaging learning pipeline
    sd_iter: number of time experiment will be repeated
    iteration: which iteration is currently based on this the data will be fetched from file
"""
for i in range(cfg.sd_iter):
    F = FederatedAveragingServer(f'{date_time_str}__it-{i}__fed_a', i)
    log_paths.append(F.log.logging_path)
end_time = datetime.now()

print(f" central start time: {start_time} Runtime: {end_time - start_time}")
print(f"log_paths: {log_paths}")