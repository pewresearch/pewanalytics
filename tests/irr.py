from __future__ import print_function
import unittest


class IRRTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd

        self.dataset = pd.DataFrame(
            {
                "code": [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                "same_codes": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                "no_variation_codes": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "coder": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "weight": [0.5, 1.5, 1.0, 0.75, 0.175, 0.5, 1.5, 1.0, 0.75, 0.175],
                "doc": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            }
        )

    def test_kappa_sample_size_power(self):
        from pewanalytics.stats.irr import kappa_sample_size_power

        sample_size = kappa_sample_size_power(
            0.4, 0.5, 0.8, 0.7, alpha=0.05, power=0.8, twosided=False
        )
        self.assertTrue(sample_size == 271)
        sample_size = kappa_sample_size_power(
            0.4, 0.5, 0.8, 0.7, alpha=0.05, power=0.8, twosided=True
        )
        self.assertTrue(sample_size == 349)

    def test_kappa_sample_size_CI(self):
        from pewanalytics.stats.irr import kappa_sample_size_CI

        sample_size = kappa_sample_size_CI(0.8, 0.7, 0.5, kappaU=None, alpha=0.05)
        self.assertTrue(sample_size == 140)
        sample_size = kappa_sample_size_CI(0.8, 0.7, 0.5, kappaU=0.9, alpha=0.05)
        self.assertTrue(sample_size == 197)

    def test_compute_scores(self):
        from pewanalytics.stats.irr import compute_scores

        scores = compute_scores(
            self.dataset, 1, 2, "code", "doc", "coder", "weight", pos_label=1
        )
        self.assertAlmostEqual(scores["roc_auc"], 0.83, 2)
        self.assertAlmostEqual(scores["matthews_corrcoef"], 0.68, 2)
        self.assertAlmostEqual(scores["alpha_unweighted"], 0.64, 2)
        self.assertAlmostEqual(scores["f1"], 0.8, 2)
        self.assertAlmostEqual(scores["recall"], 0.67, 2)
        self.assertAlmostEqual(scores["coder1_std"], 0.29, 2)
        self.assertAlmostEqual(scores["coder2_std"], 0.28, 2)
        self.assertAlmostEqual(scores["coder1_std_unweighted"], 0.24, 2)
        self.assertAlmostEqual(scores["coder2_std_unweighted"], 0.24, 2)
        self.assertAlmostEqual(scores["precision"], 1.0, 2)
        self.assertAlmostEqual(scores["n"], 5, 2)
        self.assertAlmostEqual(scores["cohens_kappa"], 0.63, 2)
        self.assertAlmostEqual(scores["precision_recall_min"], 0.67, 2)
        self.assertAlmostEqual(scores["coder2_mean"], 0.38, 2)
        self.assertAlmostEqual(scores["coder1_mean"], 0.57, 2)
        self.assertAlmostEqual(scores["coder2_mean_unweighted"], 0.4, 2)
        self.assertAlmostEqual(scores["coder1_mean_unweighted"], 0.6, 2)
        self.assertAlmostEqual(scores["pct_agree_unweighted"], 0.8, 2)
        self.assertAlmostEqual(scores["accuracy"], 0.81, 2)

        scores = compute_scores(
            self.dataset, 1, 2, "same_codes", "doc", "coder", "weight", pos_label=1
        )
        self.assertAlmostEqual(scores["roc_auc"], 1.0, 2)
        self.assertAlmostEqual(scores["matthews_corrcoef"], 1.0, 2)
        self.assertAlmostEqual(scores["alpha_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["f1"], 1.0, 2)
        self.assertAlmostEqual(scores["recall"], 1.0, 2)
        self.assertAlmostEqual(scores["coder1_std"], 0.284, 2)
        self.assertAlmostEqual(scores["coder2_std"], 0.284, 2)
        self.assertAlmostEqual(scores["coder1_std_unweighted"], 0.245, 2)
        self.assertAlmostEqual(scores["coder2_std_unweighted"], 0.245, 2)
        self.assertAlmostEqual(scores["precision"], 1.0, 2)
        self.assertAlmostEqual(scores["n"], 5, 2)
        self.assertAlmostEqual(scores["cohens_kappa"], 1.0, 2)
        self.assertAlmostEqual(scores["precision_recall_min"], 1.0, 2)
        self.assertAlmostEqual(scores["coder2_mean"], 0.382, 2)
        self.assertAlmostEqual(scores["coder1_mean"], 0.382, 2)
        self.assertAlmostEqual(scores["coder2_mean_unweighted"], 0.4, 2)
        self.assertAlmostEqual(scores["coder1_mean_unweighted"], 0.4, 2)
        self.assertAlmostEqual(scores["pct_agree_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["accuracy"], 1.0, 2)

        scores = compute_scores(
            self.dataset,
            1,
            2,
            "no_variation_codes",
            "doc",
            "coder",
            "weight",
            pos_label=1,
        )
        self.assertIsNone(scores["roc_auc"])
        self.assertAlmostEqual(scores["matthews_corrcoef"], 1.0, 2)
        self.assertAlmostEqual(scores["alpha_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["f1"], 1.0, 2)
        self.assertAlmostEqual(scores["recall"], 1.0, 2)
        self.assertAlmostEqual(scores["coder1_std"], 0.0, 2)
        self.assertAlmostEqual(scores["coder2_std"], 0.0, 2)
        self.assertAlmostEqual(scores["coder1_std_unweighted"], 0.0, 2)
        self.assertAlmostEqual(scores["coder2_std_unweighted"], 0.0, 2)
        self.assertAlmostEqual(scores["precision"], 1.0, 2)
        self.assertAlmostEqual(scores["n"], 5, 2)
        self.assertAlmostEqual(scores["cohens_kappa"], 1.0, 2)
        self.assertAlmostEqual(scores["precision_recall_min"], 1.0, 2)
        self.assertAlmostEqual(scores["coder2_mean"], 1.0, 2)
        self.assertAlmostEqual(scores["coder1_mean"], 1.0, 2)
        self.assertAlmostEqual(scores["coder2_mean_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["coder1_mean_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["pct_agree_unweighted"], 1.0, 2)
        self.assertAlmostEqual(scores["accuracy"], 1.0, 2)

    def test_compute_overall_scores(self):
        from pewanalytics.stats.irr import compute_overall_scores

        scores = compute_overall_scores(self.dataset, "doc", "code", "coder")
        self.assertAlmostEqual(scores["alpha"], 0.64, 2)
        self.assertAlmostEqual(scores["fleiss_kappa"], 0.6, 2)

    def tearDown(self):
        pass
