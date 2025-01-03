from datetime import datetime

from configuration import Configuration
from pytorch_image_classification.centralised_learning import CentralLearning

start_time = datetime.now()  # Format the date and time as a string
date_time_str = start_time.strftime("%Y-%m-%d%H-%M-%S")
log_paths = []
cfg = Configuration()
for i in range(cfg.sd_iter):
    c = CentralLearning(identifier=f'{date_time_str}__it-{i}__central', iteration=i)
    c.train_central_basic(c.train, c.val, c.test)
    log_paths.append(c.log.logging_path)
end_time = datetime.now()

print(f" central start time: {start_time} Runtime: {end_time - start_time}")
print(f"log_paths: {log_paths}")
