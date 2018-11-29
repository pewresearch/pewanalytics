from __future__ import print_function
import unittest


class NERTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_namedentityextractor(self):
        from pewanalytics.text.ner import NamedEntityExtractor
        ner = NamedEntityExtractor("This extractor is a wrapper around NLTK.")
        entities = ner.extract()
        self.assertTrue("NLTK" in entities["ORGANIZATION"])

    def tearDown(self):
        pass