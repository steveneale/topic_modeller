#!/usr/bin/env python3
"""
'tokenise.py'

Tokenise text using NLTK

2019 Steve Neale <steveneale3000@gmail.com>

"""

from nltk.tokenize import sent_tokenize, word_tokenize


def tokenise(text):
    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]
