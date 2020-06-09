from __future__ import print_function
import unittest
import pandas as pd
import math
import os
import copy
import random


class StatsMutualInfoTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df["sentiment"] = self.df["fileid"].map(lambda x: x.split("/")[0])
        self.doc = self.df["text"].values[0]
        random.seed(42)

    def test_mutual_info(self):

        from pewanalytics.text import TextDataFrame
        from pewanalytics.stats.mutual_info import compute_mutual_info

        self.df["outcome"] = (self.df["sentiment"] == "pos").astype(int)
        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        tdf.corpus["weight"] = 1.0

        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf, weights=None
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf, weights=tdf.corpus["weight"]
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf, normalize=False
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(tdf.corpus["outcome"], tdf.tfidf, l=1)
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf.todense(), weights=None
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf.todense(), weights=tdf.corpus["weight"]
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf.todense(), normalize=False
        )
        self.assertIsNotNone(mutual_info)
        mutual_info = compute_mutual_info(
            tdf.corpus["outcome"], tdf.tfidf.todense(), l=1
        )
        self.assertIsNotNone(mutual_info)

        # TODO: it would be good to test the weights and other parameter combinations at some point
        # For now, this just makes sure that everything runs
        # The more extensive mutual information tests are in `text.py` under `test_tdf_mutual_info`

    def test_mutual_info_scatter_plot(self):

        from pewanalytics.text import TextDataFrame
        from pewanalytics.stats.mutual_info import mutual_info_scatter_plot
        import matplotlib.pyplot as plt

        self.df["outcome"] = (self.df["sentiment"] == "pos").astype(int)
        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        mutual_info = tdf.mutual_info("outcome")
        plot = mutual_info_scatter_plot(
            mutual_info,
            filter_col="MI1",
            top_n=20,
            x_col="pct_term_pos_neg_ratio",
            scale_x_even=True,
            y_col="MI1",
            scale_y_even=True,
        )
        # plt.show()
        # self.assertEqual(str(plot.__hash__()), '308194536')
        # TODO: figure out how to get a unique representation of the plot
        self.assertTrue(True)

    def test_mutual_info_bar_plot(self):

        from pewanalytics.text import TextDataFrame
        from pewanalytics.stats.mutual_info import mutual_info_bar_plot
        import matplotlib.pyplot as plt

        self.df["outcome"] = (self.df["sentiment"] == "pos").astype(int)
        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        mutual_info = tdf.mutual_info("outcome")
        plot = mutual_info_bar_plot(
            mutual_info,
            filter_col="pct_term_pos_neg_ratio",
            top_n=20,
            x_col="pct_term_pos_neg_ratio",
        )
        # plt.show()
        # self.assertEqual(str(plot.__hash__()), '-9223372036574337697')
        # TODO: figure out how to get a unique representation of the plot
        self.assertTrue(True)

    def tearDown(self):
        pass
