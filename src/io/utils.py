#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'utils.py' (topic_modeller/src/io)

Input/output utility functions

2019 Steve Neale <steveneale3000@gmail.com>

"""

import os
import pickle

import pandas as pd


def load_from_file(file_path, split_lines=True, strip_lines=False):
    with open(file_path, "r", encoding="utf-8") as loaded_file:
        if split_lines:
            return [line.strip() for line in loaded_file.read().splitlines()] if strip_lines \
                   else loaded_file.read().splitlines()
        else:
            return loaded_file.read()


def load_data_frame_from_csv(file_path, seperator=",", header=0):
    return pd.read_csv(file_path, sep=seperator, header=header)


def save_pickle_object_to_file(object_to_pickle, file_path):
    pickle.dump(object_to_pickle, open(file_path, "wb"), protocol=4)


def load_pickled_object(file_path):
    return pickle.load(open(file_path, "rb"))


def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path
