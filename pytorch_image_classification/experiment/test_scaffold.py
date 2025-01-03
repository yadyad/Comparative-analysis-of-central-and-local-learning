from datetime import datetime

from configuration import Configuration
from pytorch_image_classification.federated_learning.scaffold.serverscaffold import SCAFFOLD

cfg = Configuration()
start_time = datetime.now()  # Format the date and time as a string
date_time_str = start_time.strftime("%Y-%m-%d%H-%M-%S")
log_paths = []
"""
    code for running experiment on SCAFFOLD learning pipeline
    sd_iter: number of time experiment will be repeated
    i: which iteration is currently based on this the data will be fetched from file
"""
for i in range(cfg.sd_iter):
    S = SCAFFOLD(f'{date_time_str}__it-{i}__fed_s', i)
    log_paths.append(S.log.logging_path)
end_time = datetime.now()

print(f" central start time: {start_time} Runtime: {end_time - start_time}")
print(f"log_paths: {log_paths}")