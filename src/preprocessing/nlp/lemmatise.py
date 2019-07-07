#!/usr/bin/env python3
"""
'lemmatise.py'

Lemmatise words using NLTK

2019 Steve Neale <steveneale3000@gmail.com>

"""

from nltk.stem import WordNetLemmatizer


def lemmatise(word, pos):
    return WordNetLemmatizer().lemmatize(word, pos=pos)
