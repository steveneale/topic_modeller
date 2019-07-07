#!/usr/bin/env python3
"""
'model_io.py'

Model I/O class

2019 Steve Neale <steveneale3000@gmail.com>

"""

import logging

from src.io.utils import create_directory, save_pickle_object_to_file


class ModelIO:

    def __init__(self, logging_level=logging.INFO):
        self.logger = self.setup_logger(logging_level=logging_level)

    def save_model(self, model_object, model_name):
        output_directory = create_directory("output/models/{}".format(model_name))
        save_pickle_object_to_file(model_object.model, "{}/{}".format(output_directory, "topic_model.pkl"))
        save_pickle_object_to_file(model_object.vectoriser, "{}/{}".format(output_directory, "vectoriser.pkl"))
        self.logger.info("Topic model and vectoriser saved to '{}/{}'".format(output_directory, model_name))

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
