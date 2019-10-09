from __future__ import print_function
import unittest


class NERTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_namedentityextractor(self):
        from pewanalytics.text.ner import NamedEntityExtractor

        ner = NamedEntityExtractor()
        entities = ner.extract("This extractor is a wrapper around NLTK.")
        self.assertTrue("NLTK" in entities["ORGANIZATION"])

    def tearDown(self):
        pass
