from __future__ import print_function
import unittest


class TextDatesTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_datefinder(self):
        import datetime
        from pewanalytics.text.dates import DateFinder

        finder = DateFinder()
        text = "January 1, 2018 and 02/01/2019 and Mar. 1st 2020"
        low_bound = datetime.datetime(2017, 1, 1)
        high_bound = datetime.datetime(2021, 1, 1)
        dates = finder.find_dates(text, low_bound, high_bound)
        dates = [d[0] for d in dates]
        self.assertTrue(datetime.datetime(2018, 1, 1) in dates)
        self.assertTrue(datetime.datetime(2019, 2, 1) in dates)
        self.assertTrue(datetime.datetime(2020, 3, 1) in dates)

    def tearDown(self):
        pass
