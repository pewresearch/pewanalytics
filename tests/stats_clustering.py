from __future__ import print_function
import unittest
import pandas as pd
import math
import os
import copy
import random


class StatsClusteringTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df["sentiment"] = self.df["fileid"].map(lambda x: x.split("/")[0])
        self.doc = self.df["text"].values[0]
        random.seed(42)

    def test_compute_kmeans_clusters(self):
        from pewanalytics.stats.clustering import compute_kmeans_clusters
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        kmeans = compute_kmeans_clusters(tdf.tfidf, k=2, return_score=False)
        self.assertEqual(len(kmeans), 2000)
        self.assertEqual(len(set(kmeans)), 2)
        kmeans, score = compute_kmeans_clusters(tdf.tfidf, k=2, return_score=True)
        self.assertEqual(len(kmeans), 2000)
        self.assertEqual(len(set(kmeans)), 2)
        self.assertGreater(score, 0)

    def test_compute_hdbscan_clusters(self):
        from pewanalytics.stats.clustering import compute_hdbscan_clusters
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        hdbscan = compute_hdbscan_clusters(tdf.tfidf, min_cluster_size=10)
        self.assertEqual(len(hdbscan), 2000)
        self.assertEqual(len(set(hdbscan)), 23)

    def tearDown(self):
        pass
