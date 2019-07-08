#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'test_utils.py'

Unit tests for 'utils.py'

2019 Steve Neale <steveneale3000@gmail.com>
"""

import unittest

import pandas as pd

from src.io.utils import load_data_frame_from_csv


class TestUtils(unittest.TestCase):

    def test_load_data_frame_from_csv(self):
    	data_frame = load_data_frame_from_csv("tests/fixtures/test.csv")
    	self.assertEqual(type(data_frame), pd.DataFrame)
    	self.assertEqual(data_frame.columns.tolist(), ["publish_date", "headline_text"])
    	self.assertEqual(data_frame.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()