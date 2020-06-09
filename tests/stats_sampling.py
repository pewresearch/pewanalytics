from __future__ import print_function
import unittest
import os
import pandas as pd


class StatsSamplingTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df["id"] = self.df.index
        self.df["flag"] = self.df["text"].map(lambda x: "disney" in x).astype(int)
        self.df["flag2"] = self.df["text"].map(lambda x: "princess" in x).astype(int)
        self.df["flag3"] = 0
        self.df.loc[self.df.index == 0, "flag3"] = 1
        self.oversample = pd.concat(
            [self.df[self.df["flag"] == 1.0][:50], self.df[self.df["flag"] == 0.0][:50]]
        )
        self.sample = self.df[:100]
        self.frame_props = (self.df.groupby(["flag", "flag2"]).count() / len(self.df))[
            "id"
        ].to_dict()

    def test_compute_sample_weights_from_frame(self):
        from pewanalytics.stats.sampling import compute_sample_weights_from_frame

        weights = compute_sample_weights_from_frame(self.df, self.oversample, ["flag"])
        weights = sorted([round(w, 3) for w in weights.unique()])
        self.assertEqual([0.096, 1.904], weights)

    def test_compute_balanced_sample_weights(self):
        from pewanalytics.stats.sampling import compute_balanced_sample_weights

        weights = compute_balanced_sample_weights(self.sample, ["flag"])
        self.sample["temp"] = weights
        weights = sorted([round(w, 3) for w in weights.unique()])
        self.assertEqual([0.515, 16.667], weights)
        for val in self.sample.groupby("flag").sum()["temp"].values:
            self.assertAlmostEqual(val, 50, 1)

    def test_sample_extractor_all(self):
        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        all = SampleExtractor(self.df, id_col="id", seed=42).extract(
            None, sampling_strategy="all"
        )
        self.assertTrue(len(all) == len(self.df))

    def test_sample_extractor_random(self):
        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        random = SampleExtractor(self.df, "id", seed=42).extract(
            100, sampling_strategy="random"
        )
        self.assertEqual(len(random), 100)
        self.assertEqual(self.df.loc[random]["flag"].mean(), 0.03)
        self.assertEqual(self.df["flag"].mean(), 0.048)

    def test_sample_extractor_stratify(self):
        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        normal_strat = SampleExtractor(self.df, "id", seed=42).extract(
            100, stratify_by=["flag", "flag2"], sampling_strategy="stratify"
        )
        self.assertTrue(len(normal_strat) == 100)
        sample_props = (
            self.df.loc[normal_strat].groupby(["flag", "flag2"]).count()
            / len(normal_strat)
        )["id"].to_dict()
        for k, v in self.frame_props.items():
            self.assertTrue(k in sample_props.keys())
            self.assertGreater(sample_props[k], 0)
        self.assertEqual(sample_props[(0, 1)], 0.02)
        self.assertEqual(sample_props[(1, 0)], 0.04)
        self.assertEqual(sample_props[(0, 0)], 0.93)
        self.assertEqual(sample_props[(1, 1)], 0.01)

        normal_strat = SampleExtractor(self.df, "id", seed=42).extract(
            100, stratify_by=["flag", "flag2", "flag3"], sampling_strategy="stratify"
        )
        self.assertTrue(len(normal_strat) == 100)
        sample_props = (
            self.df.loc[normal_strat].groupby(["flag", "flag2", "flag3"]).count()
            / len(normal_strat)
        )["id"].to_dict()
        for val1, val2, val3 in sample_props.keys():
            self.assertNotEqual(val3, 1)
        self.assertEqual(sample_props[(1, 0, 0)], 0.04)
        self.assertEqual(sample_props[(1, 1, 0)], 0.01)
        self.assertEqual(sample_props[(0, 1, 0)], 0.02)
        self.assertEqual(sample_props[(0, 0, 0)], 0.93)

    def test_sample_extractor_stratify_even(self):
        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        even = SampleExtractor(self.df, "id", seed=42).extract(
            20, sampling_strategy="stratify_even", stratify_by=["flag", "flag2"]
        )
        self.assertEqual(len(even), 20)
        sample_props = (
            self.df.loc[even].groupby(["flag", "flag2"]).count() / len(even)
        )["id"].to_dict()
        for val in sample_props.values():
            self.assertEqual(val, 0.25)

    def test_sample_extractor_stratify_guaranteed(self):
        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        guaranteed = SampleExtractor(self.df, "id", seed=42).extract(
            100, sampling_strategy="stratify_guaranteed", stratify_by=["flag", "flag2"]
        )
        self.assertEqual(len(guaranteed), 100)
        sample_props = (
            self.df.loc[guaranteed].groupby(["flag", "flag2"]).count() / len(guaranteed)
        )["id"].to_dict()
        for k, v in self.frame_props.items():
            self.assertTrue(k in sample_props.keys())
            self.assertGreater(sample_props[k], 0)
        self.assertEqual(sample_props[(0, 1)], 0.03)
        self.assertEqual(sample_props[(1, 0)], 0.05)
        self.assertEqual(sample_props[(0, 0)], 0.91)
        self.assertEqual(sample_props[(1, 1)], 0.01)

        guaranteed = SampleExtractor(self.df, "id", seed=42).extract(
            4, sampling_strategy="stratify_guaranteed", stratify_by=["flag", "flag2"]
        )
        self.assertEqual(len(guaranteed), 4)
        sample_props = (
            self.df.loc[guaranteed].groupby(["flag", "flag2"]).count() / len(guaranteed)
        )["id"].to_dict()
        for val in sample_props.values():
            self.assertEqual(val, 0.25)

    def tearDown(self):
        pass
