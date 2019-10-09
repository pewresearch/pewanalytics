from __future__ import print_function
import unittest
import pandas as pd
import os


class TopicsTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))[:200]
        self.doc = self.df["text"].values[0]

    def test_scikit_lda_topic_model(self):
        from pewanalytics.text.topics import ScikitLDATopicModel

        model = ScikitLDATopicModel(
            self.df, "text", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def test_scikit_nmf_topic_model(self):
        from pewanalytics.text.topics import ScikitNMFTopicModel

        model = ScikitNMFTopicModel(
            self.df, "text", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def test_gensim_lda_topic_model(self):
        from pewanalytics.text.topics import GensimLDATopicModel

        model = GensimLDATopicModel(
            self.df, "text", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def test_gensim_lda_topic_model_multicore(self):
        from pewanalytics.text.topics import GensimLDATopicModel

        model = GensimLDATopicModel(
            self.df, "text", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit(use_multicore=True)
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def test_gensim_hdp_topic_model(self):
        from pewanalytics.text.topics import GensimHDPTopicModel

        model = GensimHDPTopicModel(self.df, "text", min_df=10, max_df=0.8)
        model.fit()
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def test_corex_topic_model(self):
        from pewanalytics.text.topics import CorExTopicModel

        model = CorExTopicModel(self.df, "text", num_topics=5, min_df=10, max_df=0.8)
        model.fit()
        scores = model.get_score()
        features = model.get_features(self.df)
        topics = model.get_topics()
        doc_topics = model.get_document_topics(self.df)
        self.assertTrue(True)

    def tearDown(self):
        pass
