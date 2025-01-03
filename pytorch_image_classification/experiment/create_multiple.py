from configuration import Configuration
from data_preprocessing import DataPreprocessing

cfg = Configuration()
for i in range(cfg.sd_iter):
    D = DataPreprocessing(i)
