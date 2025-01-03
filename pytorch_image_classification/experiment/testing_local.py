from configuration import Configuration
from pytorch_image_classification.local_learning_pytorch import LearningLocal
from datetime import datetime
cfg = Configuration()
start_time = datetime.now()  # Format the date and time as a string
date_time_str = start_time.strftime("%Y-%m-%d%H-%M-%S")
log_paths = []
for i in range(cfg.sd_iter):
    l = LearningLocal(identifier=f'{date_time_str}__it-{i}__local', iteration=i)
    l.train_model(l.train_parts, l.val_parts, l.test_parts)
    log_paths.append(l.log.logging_path)
end_time = datetime.now()

print(f"start time: {start_time} Runtime: {end_time - start_time}")
print(f"log_paths: {log_paths}")
