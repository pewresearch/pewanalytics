from __future__ import print_function
import unittest
import pandas as pd
import os


class TextTopicsTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))[:200]
        self.doc = self.df["text"].values[0]

    def test_scikit_lda_topic_model(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(
            self.df, "text", "sklearn_lda", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        self.assertIn("perplexity", scores.keys())
        self.assertIn("score", scores.keys())
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        self.assertEqual(len(topics), 5)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, 5))

    def test_scikit_nmf_topic_model(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(
            self.df, "text", "sklearn_nmf", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        self.assertEqual(len(scores.keys()), 0)
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        self.assertEqual(len(topics), 5)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, 5))

    def test_gensim_lda_topic_model(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(
            self.df, "text", "gensim_lda", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        self.assertEqual(len(scores.keys()), 0)
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        self.assertEqual(len(topics), 5)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, 5))

    def test_gensim_lda_topic_model_multicore(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(
            self.df, "text", "gensim_lda", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit(use_multicore=True)
        scores = model.get_score()
        self.assertEqual(len(scores.keys()), 0)
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        self.assertEqual(len(topics), 5)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, 5))

    def test_gensim_hdp_topic_model(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(self.df, "text", "gensim_hdp", min_df=10, max_df=0.8)
        model.fit()
        scores = model.get_score()
        self.assertEqual(len(scores.keys()), 0)
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        num_topics = len(topics)
        self.assertGreater(num_topics, 0)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, num_topics))

    def test_corex_topic_model(self):
        from pewanalytics.text.topics import TopicModel

        model = TopicModel(
            self.df, "text", "corex", num_topics=5, min_df=10, max_df=0.8
        )
        model.fit()
        scores = model.get_score()
        self.assertIn("total_correlation", scores.keys())
        features = model.get_features(self.df)
        self.assertEqual(
            features.shape, (200, len(model.vectorizer.get_feature_names()))
        )
        topics = model.get_topics()
        self.assertEqual(len(topics), 5)
        doc_topics = model.get_document_topics(self.df)
        self.assertEqual(doc_topics.shape, (200, 5))

    def tearDown(self):
        pass
