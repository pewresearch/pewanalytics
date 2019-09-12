from __future__ import print_function
import unittest
import pandas as pd
import os


class TextTests(unittest.TestCase):

    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.doc = self.df['text'].values[0]

    def test_filter_parts_of_speech(self):
        from pewanalytics.text import filter_parts_of_speech
        filter_parts_of_speech(self.doc)
        self.assertTrue(True)

    def test_sentence_tokenizer(self):
        from pewanalytics.text import SentenceTokenizer
        tokenizer = SentenceTokenizer()
        tokenizer.tokenize(self.doc, min_length=10)
        self.assertTrue(True)

    def test_text_overlap_extractor(self):
        from pewanalytics.text import TextOverlapExtractor
        extractor = TextOverlapExtractor()
        overlaps = extractor.get_text_overlaps(
            self.df['text'].values[0],
            self.df['text'].values[1]
        )
        self.assertTrue(True)

    def test_text_cleaner(self):
        from pewanalytics.text import TextCleaner
        cleaner = TextCleaner()
        cleaner.clean(self.doc)
        self.assertTrue(True)

    def test_tdf_search_corpus(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text')
        tdf.search_corpus("movie")
        self.assertTrue(True)

    def test_tdf_extract_corpus_fragments(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df[:100], 'text')
        tdf.extract_corpus_fragments(
            scan_top_n_matches_per_doc=1,
            min_fragment_length=10
        )
        self.assertTrue(True)

    def test_tdf_find_duplicates(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df[:100], 'text')
        tdf.find_duplicates(
            tfidf_threshold=.9,
            fuzzy_ratio_threshold=90,
            allow_partial=False
        )
        self.assertTrue(True)

    def test_tdf_mutual_info(self):
        import random
        from pewanalytics.text import TextDataFrame
        self.df['outcome'] = self.df['text'].map(lambda x: random.choice([0, 1]))
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.mutual_info("outcome")
        self.assertTrue(True)

    def test_tdf_kmeans_clusters(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.kmeans_clusters(k=2)
        tdf.top_cluster_terms("kmeans")
        self.assertTrue(True)

    def test_tdf_hdbscan_clusters(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.hdbscan_clusters(min_cluster_size=10)
        tdf.top_cluster_terms("hdbscan")
        self.assertTrue(True)

    def test_tdf_pca_components(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.pca_components(k=5)
        tdf.get_top_documents(component_prefix="pca", top_n=1)
        self.assertTrue(True)

    def test_tdf_lsa_components(self):
        from pewanalytics.text import TextDataFrame
        tdf = TextDataFrame(self.df, 'text', min_frequency=50, max_df=.5)
        tdf.lsa_components(k=5)
        tdf.get_top_documents(component_prefix="lsa", top_n=1)
        self.assertTrue(True)

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

    def tearDown(self):
        pass