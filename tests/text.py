from __future__ import print_function
import unittest
import pandas as pd
import math
import os
import copy
import re
import random


class TextTests(unittest.TestCase):
    def setUp(self):

        self.df = pd.read_csv(os.path.join("tests", "test_data.csv"))
        self.df["sentiment"] = self.df["fileid"].map(lambda x: x.split("/")[0])
        self.doc = self.df["text"].values[0]
        random.seed(42)

    def test_tdf_find_related_keywords(self):

        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(
            self.df,
            "text",
            min_df=10,
            max_df=0.95,
            use_idf=False,
            binary=True,
            sublinear_tf=False,
            smooth_idf=False,
            norm=None,
        )
        terms = tdf.find_related_keywords("disney", n=25)
        for term in ["animation", "mulan", "mermaid", "hercules", "tarzan", "pixar"]:
            self.assertIn(term, terms)

    def test_has_fragment(self):
        from pewanalytics.text import has_fragment

        self.assertTrue(has_fragment("testing one two three", "one two"))
        self.assertFalse(has_fragment("testing one two three", "four"))

    def test_remove_fragments(self):
        from pewanalytics.text import remove_fragments

        for val, frags, expected in [
            ("testing one two three", ["one two"], "testing  three"),
            ("testing one two three", ["testing", "three"], " one two "),
        ]:
            self.assertEqual(remove_fragments(val, frags), expected)

    def test_filter_parts_of_speech(self):
        from pewanalytics.text import filter_parts_of_speech

        for pos, expected in [
            ("CC", "and but and and but"),
            ("DT", "a an the the the a the a a this an"),
            ("IN", "into of in for on in since"),
            ("JJ", "mind-fuck teen cool bad"),
            (
                "NN",
                "plot teen church party drink drive accident guys girlfriend life deal movie critique movie generation idea package review i",
            ),
            ("NNS", "couples nightmares films"),
            ("PRP", "they him it"),
            ("RB", "then very very even generally"),
            ("RBR", "harder"),
            ("VB", "see what's watch write"),
            ("VBP", "go get sorta find applaud attempt"),
            ("VBZ", "dies continues has touches presents is makes"),
            ("WP", "what"),
        ]:
            filtered = filter_parts_of_speech(self.doc[:500], filter_pos=[pos])
            self.assertEqual(filtered, expected)

        result = filter_parts_of_speech(self.doc[:100], filter_pos=None)
        self.assertEqual(result, "plot teen church party drink drive accident")
        result = filter_parts_of_speech(self.doc[:100], exclude=True)
        self.assertEqual(
            result, ": two couples go to a , and then . they get into an ."
        )

    def test_sentence_tokenizer(self):
        from pewanalytics.text import SentenceTokenizer

        tokenizer = SentenceTokenizer()
        tokenized = tokenizer.tokenize(self.doc[:500], min_length=10)
        expected = [
            "plot : two teen couples go to a church party , drink and then drive .",
            "they get into an accident .",
            "one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares .",
            "what's the deal ?",
            'watch the movie and " sorta " find out .',
            "critique : a mind-fuck movie for the teen generation that touches on a very cool idea , but presents it in a very bad package .",
            "which is what makes this review an even harder one to write , since i generally applaud films which attempt",
        ]
        for expected, tokenized in zip(expected, tokenized):
            self.assertEqual(expected, tokenized)
        regex = re.compile(r"\:")
        tokenizer = SentenceTokenizer(
            regex_split_trailing=regex, regex_split_leading=regex
        )
        tokenized = tokenizer.tokenize(self.doc[:100], min_length=10)
        expected = [
            "two teen couples go to a church party , drink and then drive .",
            "they get into an accident .",
        ]
        for expected, tokenized in zip(expected, tokenized):
            self.assertEqual(expected, tokenized)

    def test_text_overlap_extractor(self):
        from pewanalytics.text import TextOverlapExtractor

        extractor = TextOverlapExtractor()
        overlaps = extractor.get_text_overlaps(
            self.df["text"].values[0], self.df["text"].values[0][100:200]
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(
            overlaps[0],
            "one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares .",
        )

        text1 = "This is a sentence. This is another sentence. And a third sentence. And yet a fourth sentence."
        text2 = "This is a different sentence. This is another sentence. And a third sentence. But the fourth sentence is different too."
        overlaps = extractor.get_text_overlaps(
            text1, text2, tokenize=True, min_length=10
        )
        self.assertEqual(len(overlaps), 2)
        self.assertIn("This is another sentence.", overlaps)
        self.assertIn("And a third sentence.", overlaps)

        overlaps = extractor.get_text_overlaps(
            text1, text2, tokenize=False, min_length=10
        )
        self.assertEqual(len(overlaps), 2)
        self.assertIn(
            " sentence. This is another sentence. And a third sentence. ", overlaps
        )
        self.assertIn(" fourth sentence", overlaps)

        largest = extractor.get_largest_overlap(text1, text2)
        self.assertEqual(
            largest, " sentence. This is another sentence. And a third sentence. "
        )

    def test_text_cleaner(self):
        from pewanalytics.text import TextCleaner

        for params, text, expected in [
            ({}, self.doc[:100], "plot teen couple church party drink drive accident"),
            (
                {},
                "won't can't i'm ain't i'll can't wouldn't shouldn't couldn't doesn't don't i've we're i'd it's",
                "will_not would_not should_not could_not does_not do_not",
            ),
            ({"filter_pos": ["CD"], "stopwords": ["test"]}, self.doc[:100], "two"),
            (
                {"filter_pos": ["NN"]},
                self.doc[:100],
                "plot teen church party drink drive accident",
            ),
            ({"lowercase": False}, "Test One Two Three", "Test"),
            ({"remove_urls": True}, "http://www.example.com?test=asdf", ""),
            (
                {"remove_urls": False},
                "http://www.example.com?test=asdf",
                "http www example com test asdf",
            ),
            ({"strip_html": True}, "<html><body></body></html>", ""),
            (
                {"process_method": None},
                self.doc[:100],
                "plot teen couples church party drink drive accident",
            ),
            (
                {"process_method": "stem"},
                self.doc[:100],
                "plot teen coupl church parti drink drive accid",
            ),
        ]:
            cleaner = TextCleaner(**params)
            cleaned = cleaner.clean(text)
            self.assertEqual(expected, cleaned)

    def test_tdf_search_corpus(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text")
        results = tdf.search_corpus("movie")
        self.assertEqual(len(results[results["search_cosine_similarity"] > 0.2]), 5)

    def test_tdf_match_text_to_corpus(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(
            pd.DataFrame(
                [
                    {"text": "I read books"},
                    {"text": "I like reading"},
                    {"text": "I read books"},
                    {"text": "reading is nice"},
                    {"text": "reading"},
                    {"text": "books"},
                ]
            ),
            "text",
        )
        matches = tdf.match_text_to_corpus(
            ["books", "reading"], min_similarity=0.1, allow_multiple=True
        )
        self.assertEqual(
            list(matches["match_text"].values),
            ["books", "reading", "books", "reading", "reading", "books"],
        )
        matches = tdf.match_text_to_corpus(
            ["books", "reading"], min_similarity=0.5, allow_multiple=True
        )
        self.assertEqual(
            list(matches["match_text"].values),
            ["books", "reading", "books", None, "reading", "books"],
        )
        matches = tdf.match_text_to_corpus(
            ["books", "reading"], min_similarity=0.6, allow_multiple=True
        )
        self.assertEqual(
            list(matches["match_text"].values),
            ["books", None, "books", None, "reading", "books"],
        )
        matches = tdf.match_text_to_corpus(
            ["books", "reading"], min_similarity=0.5, allow_multiple=False
        )
        self.assertEqual(
            list(matches["match_text"].values),
            [None, None, None, None, "reading", "books"],
        )

    def test_tdf_extract_corpus_fragments(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df[:100], "text")
        fragments = tdf.extract_corpus_fragments(
            scan_top_n_matches_per_doc=1, min_fragment_length=3
        )
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0], "s .")

    def test_tdf_find_duplicates(self):
        from pewanalytics.text import TextDataFrame

        self.df["text"] = self.df["text"].map(lambda x: x[:1000])
        tdf = TextDataFrame(self.df, "text")
        dupes = tdf.find_duplicates(
            tfidf_threshold=0.8, fuzzy_ratio_threshold=80, allow_partial=False
        )
        self.assertEqual(len(dupes), 6)
        self.df["text"] = self.df["text"].map(
            lambda x: x[:-400] if random.random() > 0.5 else x
        )
        tdf = TextDataFrame(self.df, "text")
        dupes = tdf.find_duplicates(
            tfidf_threshold=0.6, fuzzy_ratio_threshold=80, allow_partial=True
        )
        self.assertEqual(len(dupes), 7)

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

    def test_tdf_kmeans_clusters(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        tdf.kmeans_clusters(k=2)
        terms = tdf.top_cluster_terms("kmeans")
        self.assertEqual(len(terms.keys()), 2)
        self.assertEqual(1, len(set(terms[0]).intersection(set(["husband", "alien"]))))
        self.assertEqual(1, len(set(terms[1]).intersection(set(["husband", "alien"]))))

    def test_tdf_hdbscan_clusters(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        tdf.hdbscan_clusters(min_cluster_size=10)
        terms = tdf.top_cluster_terms("hdbscan")
        self.assertEqual(len(terms.keys()), 3)
        self.assertEqual(terms[-1][0], "mike")
        self.assertEqual(terms[18][0], "disney")
        self.assertEqual(terms[11][0], "jackie")

    def test_tdf_pca_components(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        tdf.pca_components(k=5)
        docs = tdf.get_top_documents(component_prefix="pca", top_n=2)
        self.assertEqual(docs["pca_0"][0][:10], "there must")
        self.assertEqual(docs["pca_1"][0][:10], "plot : a d")
        self.assertEqual(docs["pca_2"][0][:10], "with the s")
        self.assertIn(docs["pca_3"][0][:10], ["every once", " * * * * *"])
        self.assertEqual(docs["pca_4"][0][:10], "when i fir")

    def test_tdf_lsa_components(self):
        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        tdf.lsa_components(k=5)
        docs = tdf.get_top_documents(component_prefix="lsa", top_n=2)
        self.assertEqual(docs["lsa_0"][0][:10], " * * * the")
        self.assertEqual(docs["lsa_1"][0][:10], "susan gran")
        self.assertEqual(len(docs["lsa_2"]), 0)
        self.assertIn(docs["lsa_3"][0][:10], ["as a devou", "every once"])
        self.assertEqual(docs["lsa_4"][0][:10], "when i fir")

    def test_make_word_cooccurrence_matrix(self):

        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)

        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(
            ngram_range=(1, 1), stop_words="english", min_df=10, max_df=0.5
        )
        cv.fit_transform(self.df["text"])
        vocab = cv.get_feature_names()
        mat = tdf.make_word_cooccurrence_matrix(
            normalize=False, min_frequency=10, max_frequency=0.5
        )
        self.assertTrue(len(mat) == len(vocab))
        self.assertTrue(mat.max().max() > 1.0)
        mat = tdf.make_word_cooccurrence_matrix(
            normalize=True, min_frequency=10, max_frequency=0.5
        )
        self.assertTrue(len(mat) == len(vocab))
        self.assertTrue(mat.max().max() == 1.0)

    def test_make_document_cooccurrence_matrix(self):

        from pewanalytics.text import TextDataFrame

        tdf = TextDataFrame(self.df, "text", min_df=50, max_df=0.5)
        mat = tdf.make_document_cooccurrence_matrix(normalize=False)
        self.assertTrue(len(mat) == len(self.df))
        self.assertTrue(mat.max().max() > 1.0)
        mat = tdf.make_document_cooccurrence_matrix(normalize=True)
        self.assertTrue(len(mat) == len(self.df))
        self.assertTrue(mat.max().max() == 1.0)

    def tearDown(self):
        pass
