#!/usr/bin/env python3
"""
'tag_with_pos.py'

Tag words with their part-of-speech (POS) using NLTK

2019 Steve Neale <steveneale3000@gmail.com>

"""

from nltk import pos_tag


def tag_with_pos(tokens):
    return pos_tag(tokens)
