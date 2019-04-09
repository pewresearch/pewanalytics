from __future__ import print_function
import unittest


class IRRTests(unittest.TestCase):

    def setUp(self):
        import pandas as pd
        self.dataset = pd.DataFrame({
            "code": [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
            "coder": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "weight": [.5, 1.5, 1.0, .75, .175, .5, 1.5, 1.0, .75, .175],
            "doc": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        })

    def test_kappa_sample_size_power(self):
        from pewanalytics.stats.irr import kappa_sample_size_power
        sample_size = kappa_sample_size_power(.4, .5, .8, .7, alpha=.05, power=0.8, twosided=False)
        self.assertTrue(sample_size==271)
        sample_size = kappa_sample_size_power(.4, .5, .8, .7, alpha=.05, power=0.8, twosided=True)
        self.assertTrue(sample_size==349)

    def test_kappa_sample_size_CI(self):
        from pewanalytics.stats.irr import kappa_sample_size_CI
        sample_size = kappa_sample_size_CI(.8, .7, .5, kappaU=None, alpha=.05)
        self.assertTrue(sample_size==140)
        sample_size = kappa_sample_size_CI(.8, .7, .5, kappaU=.9, alpha=.05)
        self.assertTrue(sample_size==197)

    def test_compute_scores(self):
        from pewanalytics.stats.irr import compute_scores
        scores = compute_scores(self.dataset, 1, 2, "code", "doc", "coder", "weight", pos_label=1)
        self.assertAlmostEquals(scores['roc_auc'], .83, 2)
        self.assertAlmostEquals(scores['matthews_corrcoef'], .68, 2)
        self.assertAlmostEquals(scores['alpha_unweighted'], .64, 2)
        self.assertAlmostEquals(scores['f1'], .8, 2)
        self.assertAlmostEquals(scores['recall'], .67, 2)
        self.assertAlmostEquals(scores['coder1_std'], .29, 2)
        self.assertAlmostEquals(scores['coder2_std'], .28, 2)
        self.assertAlmostEquals(scores['coder1_std_unweighted'], .24, 2)
        self.assertAlmostEquals(scores['coder2_std_unweighted'], .24, 2)
        self.assertAlmostEquals(scores['precision'], 1.0, 2)
        self.assertAlmostEquals(scores['n'], 5, 2)
        self.assertAlmostEquals(scores['cohens_kappa'], .63, 2)
        self.assertAlmostEquals(scores['precision_recall_min'], .67, 2)
        self.assertAlmostEquals(scores['coder2_mean'], .38, 2)
        self.assertAlmostEquals(scores['coder1_mean'], .57, 2)
        self.assertAlmostEquals(scores['coder2_mean_unweighted'], .4, 2)
        self.assertAlmostEquals(scores['coder1_mean_unweighted'], .6, 2)
        self.assertAlmostEquals(scores['pct_agree_unweighted'], .8, 2)
        self.assertAlmostEquals(scores['accuracy'], .81, 2)

    def test_compute_overall_scores(self):
        from pewanalytics.stats.irr import compute_overall_scores
        scores = compute_overall_scores(self.dataset, "doc", "code", "coder")
        self.assertAlmostEquals(scores['alpha'], .64, 2)
        self.assertAlmostEquals(scores['fleiss_kappa'], .6, 2)

    def tearDown(self):
        pass