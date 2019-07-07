#!/usr/bin/env python3
"""
'topic_modeller.py'

Model topics across documents using LDA (latent dirichlet allocation)

2019 Steve Neale <steveneale3000@gmail.com>

"""

import os
import logging

import nltk

from src.model import ModelBuilder
from src.io import ModelIO

root_path = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.append(os.path.join(root_path, "resources/nltk_data"))


class TopicModeller:

    def __init__(self, logging_level=logging.INFO):
        self.topic_model = None
        self.logger = self.setup_logger(logging_level=logging_level)

    def build_topic_model(self, input_path, dataset="abcnews"):
        self.logger.info("Building topic model from the '{}' dataset)".format(dataset))
        builder = ModelBuilder()
        self.topic_model = builder.build_model_from_input_data(input_path, dataset)

    def save_topic_model_with_name(self, model_name):
        if self.topic_model is None:
            raise ValueError("The .topic_model attribute is None, meaning that no model has been trained")
        ModelIO().save_model(self.topic_model, model_name)

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
