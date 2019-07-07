#!/usr/bin/env python3
"""
'language_processor.py'

Perform various (shallow) natural language processing tasks on input data

2019 Steve Neale <steveneale3000@gmail.com>

"""

import logging

from src.preprocessing.nlp import tokenise, tag_with_pos, lemmatise


class LanguageProcessor:

    def __init__(self, language="en", logging_level=logging.INFO):
        self.language = language
        self.wordnet_pos_map = {
            "NN": "n",
            "VB": "v",
            "JJ": "a",
            "RB": "r"
        }
        self.logger = self.setup_logger(logging_level=logging_level)

    def process_text(self, text):
        tokenised_text = self.tokenise_text(text)
        pos_tagged_text = self.pos_tag(tokenised_text)
        lemmatised_text = self.lemmatise_text(pos_tagged_text)
        return ". ".join([" ".join(sentence) for sentence in lemmatised_text])

    @staticmethod
    def tokenise_text(text):
        return tokenise(text)

    @staticmethod
    def pos_tag(tokenised_text):
        return [tag_with_pos(sentence) for sentence in tokenised_text]

    def lemmatise_text(self, pos_tagged_text):
        return [[lemmatise(token, self.wordnet_pos_map[pos[:2]]) if pos[:2] in ["NN", "VB", "JJ", "RB"] else token
                 for token, pos in sentence]
                for sentence in pos_tagged_text]

    @staticmethod
    def setup_logger(logging_level=logging.INFO):
        log_format = "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        return logging.getLogger(__name__)
