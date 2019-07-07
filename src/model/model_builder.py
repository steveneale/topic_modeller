#!/usr/bin/env python3
"""
'model_builder.py'

LDA-based topic model builder

2019 Steve Neale <steveneale3000@gmail.com>

"""

import logging

from progress.bar import Bar

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from src.preprocessing import DatasetLoader
from src.preprocessing.nlp import LanguageProcessor
from src.model import TopicModel


class ModelBuilder:

    def __init__(self, logging_level=logging.INFO):
        self.lda_model = None
        self.vectoriser = None
        self.logger = self.setup_logger(logging_level=logging_level)

    def build_model_from_input_data(self, input_path, dataset="abcnews_2"):
        training_data = DatasetLoader(dataset=dataset).load(input_path)
        self.logger.info("Training data loaded")
        processed_data = self.process_training_data(training_data)
        self.logger.info("Training data processed")
        bag_of_words = self.get_bag_of_words(processed_data)
        self.logger.info("Bag of words model and vocabulary created")
        self.train_lda_topic_model(bag_of_words)
        self.logger.info("LDA topic model trained")
        return TopicModel(self.lda_model, self.vectoriser)

    @staticmethod
    def process_training_data(training_data):
        processor = LanguageProcessor()
        bar = Bar("Processing training data", max=len(training_data))
        for index, text in training_data.iteritems():
            training_data[index] = processor.process_text(text)
            bar.next()
        bar.finish()
        return training_data

    def get_bag_of_words(self, training_data):
        self.vectoriser = CountVectorizer(min_df=15, max_df=0.5, max_features=100000,
                                          analyzer="word", stop_words="english")
        bag_of_words = self.vectoriser.fit_transform(training_data)
        return bag_of_words

    def train_lda_topic_model(self, bag_of_words):
        self.lda_model = LatentDirichletAllocation(n_components=10, max_iter=2, n_jobs=2, random_state=1)
        self.lda_model.fit(bag_of_words)

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
