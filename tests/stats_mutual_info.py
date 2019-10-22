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

    def test_tdf_mutual_info(self):

        from pewanalytics.text import TextDataFrame

        self.df["outcome"] = (self.df["sentiment"] == "pos").astype(int)
        self.df["text"] = self.df.apply(
            lambda x: "{} always_pos".format(x["text"]) if x["outcome"] else x["text"],
            axis=1,
        )
        tdf = TextDataFrame(
            self.df,
            "text",
            min_df=50,
            max_df=0.5,
            use_idf=False,
            binary=True,
            sublinear_tf=False,
            smooth_idf=False,
            norm=None,
        )
        # games occurs 24 times in the pos class, 26 times in the neg class; total is 50
        # overall document total is 2000 (1000 pos)
        px1y1 = 24.0 / 2000.0
        px1y0 = 26.0 / 2000.0
        px1 = 50.0 / 2000.0
        px0 = (2000.0 - 50.0) / 2000.0
        py1 = 1000.0 / 2000.0

        mutual_info = tdf.mutual_info("outcome", normalize=False)
        MI1 = math.log(px1y1 / (px1 * py1), 2)
        MI1_alt = math.log(px1y1, 2) - math.log(px1, 2) - math.log(py1, 2)
        self.assertAlmostEqual(mutual_info.loc["games"]["MI1"], MI1, 4)
        self.assertAlmostEqual(mutual_info.loc["games"]["MI1"], MI1_alt, 4)

        mutual_info = tdf.mutual_info("outcome", normalize=True)
        MI1_norm = MI1 / (-1 * math.log(px1y1, 2))
        MI1_norm_alt = (math.log(px1 * py1, 2) / math.log(px1y1, 2)) - 1.0
        self.assertAlmostEqual(mutual_info.loc["games"]["MI1"], MI1_norm, 4)
        self.assertAlmostEqual(mutual_info.loc["games"]["MI1"], MI1_norm_alt, 4)

        pos = mutual_info.sort_values("MI1", ascending=False)[:10]
        neg = mutual_info.sort_values("MI0", ascending=False)[:10]

        self.assertEqual(pos.index[0], "always_pos")
        self.assertEqual(pos.iloc[0]["MI1"], 1.0)
        self.assertEqual(pos.index[1], "outstanding")
        for field, val in [
            ("MI1", 0.178374),
            ("MI0", -0.319942),
            ("total", 68.0),
            ("total_pos_with_term", 63.0),
            ("total_neg_with_term", 5.0),
            ("total_pos_neg_with_term_diff", 58.0),
            ("pct_pos_with_term", 0.063),
            ("pct_neg_with_term", 0.005),
            ("pct_pos_neg_with_term_diff", 0.058),
            ("pct_pos_neg_with_term_ratio", 12.6),
            ("pct_term_pos", 0.926471),
            ("pct_term_neg", 0.073529),
            ("pct_term_pos_neg_diff", 0.852941),
            ("pct_term_pos_neg_ratio", 12.6),
        ]:
            self.assertAlmostEqual(pos.iloc[1][field], val, 4)

        self.assertEqual(neg.index[0], "bad")
        for field, val in [
            ("MI1", -0.195836),
            ("MI0", 0.209830),
            ("total", 773.0),
            ("total_pos_with_term", 259.0),
            ("total_neg_with_term", 514.0),
            ("total_pos_neg_with_term_diff", -255.0),
            ("pct_pos_with_term", 0.259),
            ("pct_neg_with_term", 0.514),
            ("pct_pos_neg_with_term_diff", -0.255),
            ("pct_pos_neg_with_term_ratio", 0.503891),
            ("pct_term_pos", 0.335058),
            ("pct_term_neg", 0.664942),
            ("pct_term_pos_neg_diff", -0.329884),
            ("pct_term_pos_neg_ratio", 0.503891),
        ]:
            self.assertAlmostEqual(neg.iloc[0][field], val, 4)

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
