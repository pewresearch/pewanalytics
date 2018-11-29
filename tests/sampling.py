from __future__ import print_function
import unittest


class SamplingTests(unittest.TestCase):

    def setUp(self):
        import nltk
        import pandas as pd

        nltk.download("movie_reviews")

        rows = []
        for fileid in nltk.corpus.movie_reviews.fileids():
            rows.append({"text": nltk.corpus.movie_reviews.raw(fileid)})
        self.df = pd.DataFrame(rows)
        self.df['id'] = self.df.index
        self.df['flag'] = self.df['text'].map(lambda x: "disney" in x).astype(int)
        self.df['flag2'] = self.df['text'].map(lambda x: "princess" in x).astype(int)
        self.oversample = pd.concat([
            self.df[self.df['flag'] == 1.0][:50],
            self.df[self.df['flag'] == 0.0][:50]
        ])
        self.sample = self.df[:100]

    def test_compute_sample_weights_from_frame(self):
        from pewanalytics.stats.sampling import compute_sample_weights_from_frame
        weights = compute_sample_weights_from_frame(self.df, self.oversample, ['flag'])
        weights = sorted([round(w, 3) for w in weights.unique()])
        self.assertTrue([.096, 1.904]==weights)

    def test_compute_balanced_sample_weights(self):
        from pewanalytics.stats.sampling import compute_balanced_sample_weights
        weights = compute_balanced_sample_weights(self.sample, ['flag'])
        weights = sorted([round(w, 3) for w in weights.unique()])
        self.assertTrue([.515, 16.667] == weights)

    def test_sample_extractor(self):

        import numpy as np
        from pewanalytics.stats.sampling import SampleExtractor

        all = SampleExtractor(
            id_col="id",
            sampling_strategy="all"
        ).extract(self.df, None)
        self.assertTrue(len(all)==len(self.df))

        random = SampleExtractor(
            id_col="id",
            seed=42,
            sampling_strategy="random"
        ).extract(self.df, 100)
        self.assertTrue(len(random)==100)
        self.assertTrue(self.df.loc[random]['flag'].mean() < self.df['flag'].mean()*2)


        frame_props = (self.df.groupby(['flag', 'flag2']).count() / len(self.df))['id'].to_dict()
        guaranteed = SampleExtractor(
            id_col="id",
            seed=42,
            sampling_strategy="stratify_guaranteed",
            stratify_by=["flag", "flag2"]
        ).extract(self.df, 100)
        self.assertTrue(len(guaranteed) == 100)
        sample_props = (self.df.loc[guaranteed].groupby(['flag', 'flag2']).count() / len(guaranteed))['id'].to_dict()
        for k, v in frame_props.iteritems():
            self.assertTrue(k in sample_props.keys())
            self.assertTrue(sample_props[k] > 0)
            self.assertTrue(sample_props[k] < frame_props[k] + .03)
            self.assertTrue(sample_props[k] > frame_props[k] - .03)

        guaranteed = SampleExtractor(
            id_col="id",
            seed=42,
            sampling_strategy="stratify_guaranteed",
            stratify_by=["flag", "flag2"]
        ).extract(self.df, 4)
        self.assertTrue(len(guaranteed) == 4)
        sample_props = (self.df.loc[guaranteed].groupby(['flag', 'flag2']).count() / len(guaranteed))['id'].to_dict()
        self.assertTrue(np.average(sample_props.values()) == .25)

        even = SampleExtractor(
            id_col="id",
            seed=42,
            sampling_strategy="stratify_even",
            stratify_by=["flag", "flag2"]
        ).extract(self.df, 20)
        self.assertTrue(len(even) == 20)
        sample_props = (self.df.loc[even].groupby(['flag', 'flag2']).count() / len(even))['id'].to_dict()
        self.assertTrue(np.average(sample_props.values()) == .25)

        normal_strat = SampleExtractor(
            id_col="id",
            seed=42,
            stratify_by=["flag", "flag2"],
            sampling_strategy="stratify"
        ).extract(self.df, 100)
        self.assertTrue(len(normal_strat) == 100)
        sample_props = (self.df.loc[normal_strat].groupby(['flag', 'flag2']).count() / len(normal_strat))['id'].to_dict()
        for k, v in sample_props.iteritems():
            self.assertTrue(sample_props[k] > 0)
            self.assertTrue(sample_props[k] < frame_props[k] + .01)
            self.assertTrue(sample_props[k] > frame_props[k] - .01)

        strat_alt = SampleExtractor(
            id_col="id",
            seed=42,
            stratify_by=["flag", "flag2"],
            sampling_strategy="stratify_alt"
        ).extract(self.df, 100)
        self.assertTrue(len(normal_strat) == 100)
        sample_props = (self.df.loc[strat_alt].groupby(['flag', 'flag2']).count() / len(strat_alt))['id'].to_dict()
        for k, v in sample_props.iteritems():
            self.assertTrue(sample_props[k] > 0)
            self.assertTrue(sample_props[k] < frame_props[k] + .01)
            self.assertTrue(sample_props[k] > frame_props[k] - .01)

        # TODO: tests for random seeds

    def tearDown(self):
        pass