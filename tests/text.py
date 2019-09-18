from __future__ import print_function
import unittest
import pandas as pd
import os
import copy
import random


class TextTests(unittest.TestCase):

    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df['sentiment'] = self.df['fileid'].map(lambda x: x.split("/")[0])
        self.doc = self.df['text'].values[0]
        random.seed(42)

    def test_filter_parts_of_speech(self):
        from pewanalytics.text import filter_parts_of_speech
        for pos, expected in [
            ("CC", "and but and and but"),
            ("DT", "a an the the the a the a a this an"),
            ("IN", "into of in for on in since"),
            ("JJ", "mind-fuck teen cool bad"),
            ("NN", "plot teen church party drink drive accident guys girlfriend life deal movie critique movie generation idea package review i"),
            ("NNS", "couples nightmares films"),
            ("PRP", "they him it"),
            ("RB", "then very very even generally"),
            ("RBR", "harder"),
            ("VB", "see what's watch write"),
            ("VBP", "go get sorta find applaud attempt"),
            ("VBZ", "dies continues has touches presents is makes"),
            ("WP", "what")
        ]:
            filtered = filter_parts_of_speech(self.doc[:500], filter_pos=[pos])
            self.assertEqual(filtered, expected)

    def test_sentence_tokenizer(self):
        from pewanalytics.text import SentenceTokenizer
        tokenizer = SentenceTokenizer()
        tokenized = tokenizer.tokenize(self.doc[:500], min_length=10)
        expected = [
            'plot : two teen couples go to a church party , drink and then drive .',
            'they get into an accident .',
            'one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares .',
            "what's the deal ?",
            'watch the movie and " sorta " find out .',
            'critique : a mind-fuck movie for the teen generation that touches on a very cool idea , but presents it in a very bad package .',
            'which is what makes this review an even harder one to write , since i generally applaud films which attempt'
        ]
        for expected, tokenized in zip(expected, tokenized):
            self.assertEqual(expected, tokenized)

    def test_text_overlap_extractor(self):
        from pewanalytics.text import TextOverlapExtractor
        extractor = TextOverlapExtractor()
        overlaps = extractor.get_text_overlaps(
            self.df['text'].values[0],
            self.df['text'].values[0][100:200]
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0], 'one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares .')

    def test_text_cleaner(self):
        from pewanalytics.text import TextCleaner
        for params, text, expected in [
            ({}, self.doc[:100], "plot two teen couple church party drink drive get accident"),
            ({}, "won't can't i'm ain't i'll can't wouldn't shouldn't couldn't doesn't don't i've we're i'd it's", "will_not cannot cannot would_not should_not could_not does_not do_not would"),
            ({"filter_pos": ["CD"]}, self.doc[:100], "two"),
            ({"filter_pos": ["NN"]}, self.doc[:100], "plot teen church party drink drive accident"),
            ({"lowercase": False}, "Test One Two Three", "Test One Two Three"),
            ({"remove_urls": True}, "http://www.example.com?test=asdf", ""),
            ({"remove_urls": False}, "http://www.example.com?test=asdf", "http www example com test asdf"),
            ({"strip_html": True}, "<html><body></body></html>", ""),
            ({"lemmatize": False}, self.doc[:100], "plot two teen couples church party drink drive get accident")
        ]:
            cleaner = TextCleaner(**params)
            cleaned = cleaner.clean(text)
            self.assertEqual(expected, cleaned)

    def test_tdf_search_corpus(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text')
        results = tdf.search_corpus("movie")
        self.assertEqual(len(results[results['search_cosine_similarity']>.2]), 5)

    def test_tdf_extract_corpus_fragments(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df[:100], 'text')
        fragments = tdf.extract_corpus_fragments(
            scan_top_n_matches_per_doc=1,
            min_fragment_length=3
        )
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0], 's .')

    def test_tdf_find_duplicates(self):
        from pewanalytics.text import TextDataFrame
        self.df['text'] = self.df['text'].map(lambda x: x[:1000])
        tdf = TextDataFrame(self.df, 'text')
        dupes = tdf.find_duplicates(
            tfidf_threshold=.8,
            fuzzy_ratio_threshold=80,
            allow_partial=False
        )
        self.assertEqual(len(dupes), 6)
        self.df['text'] = self.df['text'].map(lambda x: x[:-400] if random.random() > .5 else x)
        tdf = TextDataFrame(self.df, 'text')
        dupes = tdf.find_duplicates(
            tfidf_threshold=.6,
            fuzzy_ratio_threshold=80,
            allow_partial=True
        )
        self.assertEqual(len(dupes), 7)

    def test_tdf_mutual_info(self):

        from pewanalytics.text import TextDataFrame
        self.df['outcome'] = (self.df['sentiment'] == 'pos').astype(int)
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        pos, neg = tdf.mutual_info("outcome")

        self.assertEqual(pos.index[0], 'outstanding')
        for field, val in [
            ("MI1", 0.101480),
            ("MI0", -0.217881),
            ("total", 5.266579),
            ("pos_freq", 4.851582),
            ("neg_freq", 0.414997),
            ("freq_diff", 4.436586),
            ("pct_pos_with_term", 0.004852),
            ("pct_neg_with_term", 0.000415),
            ("pct_term_pos", 0.921202),
            ("pct_term_neg", 0.078798),
            ("pct_term_diff", 0.842404),
            ("pct_pos_neg_diff", 0.004437),
            ("pct_pos_multiplier", 11.690651),
            ("pct_neg_multiplier", 0.085538),
            ("MI", 0.101480),
            ("freq", 4.851582),
            ("pct_with_term", 0.004852),
            ("pct_term", 0.921202)
        ]:
            self.assertAlmostEqual(pos.iloc[0][field], val, 4)

        self.assertEqual(neg.index[0], 'worst')
        for field, val in [
            ("MI1", -0.186571),
            ("MI0", 0.108647),
            ("total", 15.958741),
            ("pos_freq", 2.247558),
            ("neg_freq", 13.711183),
            ("freq_diff", -11.463624),
            ("pct_pos_with_term", 0.002248),
            ("pct_neg_with_term", 0.013711),
            ("pct_term_pos", 0.140836),
            ("pct_term_neg", 0.859164),
            ("pct_term_diff", -0.718329),
            ("pct_pos_neg_diff", -0.011464),
            ("pct_pos_multiplier", 0.163922),
            ("pct_neg_multiplier", 6.100479),
            ("MI", 0.108647),
            ("freq", 13.711183),
            ("pct_with_term", 0.013711),
            ("pct_term", 0.859164)
        ]:
            self.assertAlmostEqual(neg.iloc[0][field], val, 4)

    def test_plot_mutual_info(self):

        from pewanalytics.text import TextDataFrame
        from pewanalytics.stats.mutual_info import plot_mutual_info
        from pewtils import get_hash
        self.df['outcome'] = (self.df['sentiment'] == 'pos').astype(int)
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        pos, neg = tdf.mutual_info("outcome")
        plot = plot_mutual_info(pos)
        self.assertEqual(get_hash(plot), '3:UvJgKVWn:UvqOW')

    def test_tdf_kmeans_clusters(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.kmeans_clusters(k=2)
        terms = tdf.top_cluster_terms("kmeans")
        self.assertEqual(len(terms.keys()), 2)
        self.assertEqual(terms[1][0], "alien")
        self.assertEqual(terms[0][0], "husband")

    def test_tdf_hdbscan_clusters(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.hdbscan_clusters(min_cluster_size=10)
        terms = tdf.top_cluster_terms("hdbscan")
        self.assertEqual(len(terms.keys()), 3)
        self.assertEqual(terms[-1][0], "mike")
        self.assertEqual(terms[18][0], "disney")
        self.assertEqual(terms[11][0], "jackie")

    def test_tdf_pca_components(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.pca_components(k=5)
        docs = tdf.get_top_documents(component_prefix="pca", top_n=1)
        self.assertEqual(docs['pca_0'][0][:10], 'there must')
        self.assertEqual(docs['pca_1'][0][:10], 'plot : a d')
        self.assertEqual(docs['pca_2'][0][:10], 'with the s')
        self.assertEqual(docs['pca_3'][0][:10], 'every once')
        self.assertEqual(docs['pca_4'][0][:10], 'when i fir')

    def test_tdf_lsa_components(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.lsa_components(k=5)
        docs = tdf.get_top_documents(component_prefix="lsa", top_n=1)
        self.assertEqual(docs['lsa_0'][0][:10], ' * * * the')
        self.assertEqual(docs['lsa_1'][0][:10], 'susan gran')
        self.assertEqual(len(docs['lsa_2']), 0)
        self.assertEqual(docs['lsa_3'][0][:10], 'as a devou')
        self.assertEqual(docs['lsa_4'][0][:10], 'when i fir')

    def test_make_word_cooccurrence_matrix(self):

        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(ngram_range=(1, 1), stop_words='english', min_df=10, max_df=.5)
        cv.fit_transform(self.df['text'])
        vocab = cv.get_feature_names()
        mat = tdf.make_word_cooccurrence_matrix(normalize=False, min_frequency=10, max_frequency=.5)
        self.assertTrue(len(mat) == len(vocab))
        self.assertTrue(mat.max().max() > 1.0)
        mat = tdf.make_word_cooccurrence_matrix(normalize=True, min_frequency=10, max_frequency=.5)
        self.assertTrue(len(mat) == len(vocab))
        self.assertTrue(mat.max().max() == 1.0)

    def test_make_document_cooccurrence_matrix(self):

        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        mat = tdf.make_document_cooccurrence_matrix(normalize=False)
        self.assertTrue(len(mat)==len(self.df))
        self.assertTrue(mat.max().max() > 1.0)
        mat = tdf.make_document_cooccurrence_matrix(normalize=True)
        self.assertTrue(len(mat) == len(self.df))
        self.assertTrue(mat.max().max() == 1.0)

    def test_correspondence_analysis(self):
        from pewanalytics.stats.dimensionality_reduction import correspondence_analysis
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        matrix = pd.DataFrame(tdf.tfidf.todense(), columns=tdf.vectorizer.get_feature_names())
        mca = correspondence_analysis(matrix)
        self.assertAlmostEqual(mca['mca_1'].values[0], .59554, 4)
        self.assertEqual(mca['node'].values[0], "over")
        self.assertAlmostEqual(mca['mca_1'].values[-1], -.4274, 4)
        self.assertEqual(mca['node'].values[-1], "red")

    def tearDown(self):
        pass