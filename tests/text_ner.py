from __future__ import print_function
import unittest
import nltk
from pewtils import flatten_list


class TextNERTests(unittest.TestCase):
    def setUp(self):

        import nltk

        nltk.download("inaugural")
        fileid = nltk.corpus.inaugural.fileids()[0]
        self.text = nltk.corpus.inaugural.raw(fileid)

    def test_namedentityextractor(self):
        from pewanalytics.text.ner import NamedEntityExtractor

        for method, num_types, num_entities in [
            ("nltk", 3, 12),
            ("spacy", 6, 12),
            ("all", 7, 22),
        ]:
            ner = NamedEntityExtractor(method=method)
            entities = ner.extract(self.text)
            self.assertEqual(len(entities.keys()), num_types)
            self.assertEqual(len(flatten_list(list(entities.values()))), num_entities)

    def tearDown(self):
        pass
