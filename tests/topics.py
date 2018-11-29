from __future__ import print_function
import unittest


class TopicsTests(unittest.TestCase):

    def setUp(self):

        import nltk
        import pandas as pd

        nltk.download("movie_reviews")

        rows = []
        for fileid in nltk.corpus.movie_reviews.fileids():
            rows.append({"text": nltk.corpus.movie_reviews.raw(fileid)})
        self.df = pd.DataFrame(rows)[:200]
        self.doc = self.df['text'].values[0]

    def test_scikit_lda_topic_model(self):
        from pewanalytics.text.topics import ScikitLDATopicModel
        model = ScikitLDATopicModel(self.df, "text", num_topics=5)
        model.fit()
        model.get_score()
        model.get_features(self.df)
        model.get_topics()
        model.get_document_topics(self.df)

    def test_scikit_nmf_topic_model(self):
        from pewanalytics.text.topics import ScikitNMFTopicModel
        model = ScikitNMFTopicModel(self.df, "text", num_topics=5)
        model.fit()
        model.get_score()
        model.get_features(self.df)
        model.get_topics()
        model.get_document_topics(self.df)

    def test_gensim_lda_topic_model(self):
        from pewanalytics.text.topics import GensimLDATopicModel
        model = GensimLDATopicModel(self.df, "text", num_topics=5)
        model.fit()
        model.get_score()
        model.get_features(self.df)
        model.print_topics()
        model.get_document_topics(self.df)

    def test_gensim_hdp_topic_model(self):
        from pewanalytics.text.topics import GensimHDPTopicModel
        model = GensimHDPTopicModel(self.df, "text", num_topics=5)
        model.fit(T=10)
        model.get_score()
        model.get_features(self.df)
        model.print_topics()
        model.get_document_topics(self.df)

    def tearDown(self):
        pass