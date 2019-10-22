from __future__ import print_function
import unittest
import pandas as pd
import math
import os
import copy
import random


class StatsDimensionalityReductionTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df["sentiment"] = self.df["fileid"].map(lambda x: x.split("/")[0])
        self.doc = self.df["text"].values[0]
        random.seed(42)

    def test_get_pca(self):
        from pewanalytics.stats.dimensionality_reduction import get_pca
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        components, results = get_pca(tdf.tfidf, k=5)
        component_means = components.mean().to_dict()
        result_means = results.mean().to_dict()
        self.assertEqual(components.shape[0], 2075)
        self.assertEqual(components.shape[1], 5)
        self.assertEqual(results.shape[0], 2000)
        self.assertEqual(results.shape[1], 6)
        self.assertAlmostEqual(component_means["pca_0"], -0.0015, 2)
        self.assertAlmostEqual(component_means["pca_1"], 0.004, 2)
        self.assertAlmostEqual(component_means["pca_2"], 0.005, 2)
        self.assertAlmostEqual(component_means["pca_3"], -0.004, 2)
        self.assertAlmostEqual(component_means["pca_4"], -0.0017, 2)
        self.assertAlmostEqual(result_means["pca_0"], 0.0, 2)
        self.assertAlmostEqual(result_means["pca_1"], 0.0, 2)
        self.assertAlmostEqual(result_means["pca_2"], 0.0, 2)
        self.assertAlmostEqual(result_means["pca_3"], 0.0, 2)
        self.assertAlmostEqual(result_means["pca_4"], 0.0, 2)

    def test_get_lsa(self):
        from pewanalytics.stats.dimensionality_reduction import get_lsa
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        components, results = get_lsa(tdf.tfidf, k=5)
        component_means = components.mean().to_dict()
        result_means = results.mean().to_dict()
        self.assertEqual(components.shape[0], 2075)
        self.assertEqual(components.shape[1], 5)
        self.assertEqual(results.shape[0], 2000)
        self.assertEqual(results.shape[1], 6)
        self.assertAlmostEqual(component_means["lsa_0"], 0.0174, 2)
        self.assertAlmostEqual(component_means["lsa_1"], -0.0002, 2)
        self.assertAlmostEqual(component_means["lsa_2"], -0.0030, 2)
        self.assertAlmostEqual(component_means["lsa_3"], -0.0011, 2)
        self.assertAlmostEqual(component_means["lsa_4"], -0.0002, 2)
        self.assertAlmostEqual(result_means["lsa_0"], 0.3025, 2)
        self.assertAlmostEqual(result_means["lsa_1"], 0.001, 2)
        self.assertAlmostEqual(result_means["lsa_2"], -0.0034, 2)
        self.assertAlmostEqual(result_means["lsa_3"], -0.0022, 2)
        self.assertAlmostEqual(result_means["lsa_4"], -0.0, 2)

    def test_correspondence_analysis(self):
        from pewanalytics.stats.dimensionality_reduction import correspondence_analysis
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        matrix = pd.DataFrame(
            tdf.tfidf.todense(), columns=tdf.vectorizer.get_feature_names()
        )
        mca = correspondence_analysis(matrix)
        self.assertAlmostEqual(mca["mca_1"].values[0], 0.59554, 4)
        self.assertEqual(mca["node"].values[0], "over")
        self.assertAlmostEqual(mca["mca_1"].values[-1], -0.4274, 4)
        self.assertEqual(mca["node"].values[-1], "red")

    def tearDown(self):
        pass
