from configuration import Configuration
from data_preprocessing import DataPreprocessing

cfg = Configuration()
"""
    code for generating multiple data frame saved to file
"""
for i in range(cfg.sd_iter):
    D = DataPreprocessing(i)
