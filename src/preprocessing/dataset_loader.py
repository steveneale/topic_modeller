#!usr/bin/env python3
"""
'dataset_loader.py' (topic_modeller/src/preprocessing)

Class for loading topic modelling datasets

2019 Steve Neale <steveneale3000@gmail.com>

"""

import importlib


class DatasetLoader:

    def __init__(self, dataset="abcnews_2"):
        self.dataset = dataset
        self.data = None

    def load(self, dataset_path, *args):
        loader_module = importlib.import_module("src.preprocessing.datasets")
        loader = getattr(loader_module, "{}Loader".format(self.dataset.title()))
        self.data = loader().load(dataset_path, *args)
        return self.data
