#!/usr/bin/env python3
"""
'topic_model.py'

Topic model class

2019 Steve Neale <steveneale3000@gmail.com>

"""

import logging


class TopicModel:

    def __init__(self, model, vectoriser, logging_level=logging.INFO):
        self.model = model
        self.vectoriser = vectoriser
        self.logger = self.setup_logger(logging_level=logging_level)

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
