#!usr/bin/env python3
"""
'abcnews_loader.py' (topic_modeller/src/preprocessing/datasets)

Class for loading the ABC News Headlines dataset for topic modelling

2019 Steve Neale <steveneale3000@gmail.com>

"""

import logging

from src.io.utils import load_data_frame_from_csv


class AbcnewsLoader:

    def __init__(self, logging_level=logging.INFO):
        self.logger = self.setup_logger(logging_level=logging_level)

    @staticmethod
    def load(dataset_path):
        data_frame = load_data_frame_from_csv(dataset_path)
        data_frame = data_frame["headline_text"]
        return data_frame

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
