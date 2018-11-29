from __future__ import print_function
import unittest


class IRRTests(unittest.TestCase):

    def setUp(self):
        pass

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

    def tearDown(self):
        pass