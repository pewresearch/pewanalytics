import re
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import wordnet

from tqdm import tqdm
from stopit import ThreadingTimeout as Timeout
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from stopit import TimeoutException

from pewtils import is_null, is_not_null
from pewtils.http import strip_html
from pewtils import decode_text as _decode_text
from pewtils.regex import URL_REGEX

from pewanalytics.stats.clustering import compute_hdbscan_clusters, compute_kmeans_clusters
from pewanalytics.stats.mutual_info import compute_mutual_info
from pewanalytics.stats.dimensionality_reduction import get_lsa, get_pca



"""
.. _text:

.. tip: Insert tip

.. autosummary::
    :toctree: _autosummary
    :template: clean.rst

    dates
    fragments
    ner
    topics
"""


def has_fragment(text, fragment):
    """
    :param text: The text to search
    :param fragment: The fragment to search for
    :return: Whether or not the text contains the fragment
    """
    if any([
        (fragment in text),
        (_decode_text(fragment) in _decode_text(text))
    ]):
        return True
    else:
        return False


def remove_fragments(text, fragments, throw_loud_fail = False):
    """
    Iteratively remove fragments from a string
    :param text: string
    :param fragments: A list of string fragments to search for and remove
    :return: The original string, minus any parts that matched the fragments provided
    """
    for f in fragments:
        new_text = text.replace(f, "")
        #if the new text is the same as previous, try decoding
        if new_text == text:
            new_text = _decode_text(text, throw_loud_fail).replace(_decode_text(f, throw_loud_fail), "")
        #if the new text is still the same as previous, then new text is None
        if new_text == text:
            new_text = None
        if new_text:
            text = new_text
    return text

def filter_parts_of_speech(text, filter_pos=None, exclude = False):
    """
    Retain words associated with parts of speech in the text if exclude = False.
    If exclude = True, exclude words associated with parts of speech.
    Default is Noun (NN), Proper Noun (NNP) and Adjective (JJ)

    :param text: string
    :param filter_pos: array of part of speech tags (default is 'NN', 'NNP', and 'JJ')
        the options here are: CD, VBN, VBG, RB
        Note: the full list of POS is here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    :return: cleaned string
    """
    if not filter_pos:
        filter_pos = ('NN','NNP','JJ')
    text = text.split()
    tagged_words = nltk.pos_tag(text)
    if exclude == False:
        valid = [word[0] for word in tagged_words if word[1] in filter_pos]
    else:
        valid = [word[0] for word in tagged_words if word[1] != filter_pos]
    return " ".join(valid)


def get_fuzzy_ratio(text1, text2, throw_loud_fail = False):
    """
    Uses Levenshtein Distance to calculate similarity of two strings.  Measures how the edit distance compares \
    to the overall length of the texts.

    :param text1: First string
    :param text2: Second string
    :param throw_loud_fail: bool; does not remove all remove nonascii characters if false
    :return: The Levenshtein ratio between the two texts
    """
    try:
        return fuzz.ratio(text1, text2)
    except (UnicodeDecodeError, UnicodeEncodeError):
        return fuzz.ratio(_decode_text(text1, throw_loud_fail), _decode_text(text2, throw_loud_fail))


def get_fuzzy_partial_ratio(text1, text2, throw_loud_fail = False, timeout=5):
    """
    Useful to calculate similarity of two strings that are of noticeably different lengths.  Allows for the possibility \
    that one text is a subset of the other; finds the largest overlap and computes the Levenshtein ratio on that.


    :param text1: First string
    :param text2: Second string
    :param timeout: The number of seconds to wait before giving up
    :param throw_loud_fail: bool; does not remove all remove nonascii characters if false
    :return: The partial Levenshtein ratio between the two texts

    :accepts kwarg timeout:
    """

    partial_ratio = None
    with Timeout(timeout, swallow_exc=True):
        try:
            partial_ratio = fuzz.partial_ratio(text1, text2)
        except (UnicodeDecodeError, UnicodeEncodeError):
            partial_ratio = fuzz.partial_ratio(_decode_text(text1, throw_loud_fail), _decode_text(text2, throw_loud_fail))
        return partial_ratio


class SentenceTokenizer(object):
    """
    Initializes a tokenizer that can be be used to break text into tokens using the `tokenize` function

    :param base_tokenizer: The tokenizer to use (default = NLTK's English Punkt tokenizer)
    :param regex_split_trailing: A compiled regex object used to define the end of sentences
    :param regex_split_leading: A compiled regex object used to define the beginning of sentences
    """
    def __init__(self, base_tokenizer=None, regex_split_trailing=None, regex_split_leading=None):

        self.base_tokenizer = base_tokenizer if base_tokenizer else nltk.data.load("tokenizers/punkt/english.pickle")
        self.regex_split_trailing = regex_split_trailing
        self.regex_split_leading = regex_split_leading

    def tokenize(self, text, throw_loud_fail = False, min_length=None):
        """
        :param text: The text to tokenize
        :param throw_loud_fail: bool; does not remove all remove nonascii characters if false
        :param min_length: The minimum acceptable length of a sentence \
            (if a token is shorter than this, it will be considered part of the preceding sentence)
        :return:
        """

        text = _decode_text(text, throw_loud_fail)

        partial_tokens = []
        token_group = []
        for t in self.base_tokenizer.tokenize(text):
            if not self.regex_split_leading:
                partial_tokens.append(t)
            else:
                leaders = self.regex_split_leading.findall(t)
                token_group = []
                for subt_lead in self.regex_split_leading.split(t):
                    if subt_lead != '':
                        token_group.append(subt_lead)
                    if len(leaders) == 0 or subt_lead not in leaders:
                        partial_tokens.append("".join(token_group))
                        token_group = []
                if len(token_group) > 0:
                    partial_tokens.append("".join([t for t in token_group if t != '']))
        if len(token_group) > 0:
            partial_tokens.append("".join([t for t in token_group if t != '']))

        if not self.regex_split_trailing:
            final_tokens = partial_tokens
        else:
            final_tokens = []
            token_group = []
            for t in partial_tokens:
                trailers = self.regex_split_trailing.findall(t)
                token_group = []
                for subt_trail in self.regex_split_trailing.split(t):
                    if subt_trail != '':
                        token_group.append(subt_trail)
                    if len(trailers) == 0 or subt_trail in trailers:
                        final_tokens.append("".join(token_group))
                        token_group = []
                if len(token_group) > 0:
                    final_tokens.append("".join([t for t in token_group if t != '']))
            if len(token_group) > 0:
                final_tokens.append("".join([t for t in token_group if t != '']))

        final_tokens = [t.strip() for t in final_tokens]

        if min_length:
            final_tokens = [f for f in final_tokens if len(f) >= min_length]

        return final_tokens


class TextOverlapExtractor(object):
    """
    :param tokenizer: The tokenizer to use (default = SentenceTokenizer())
    """
    def __init__(self, tokenizer=None):

        if not tokenizer:
            self.tokenizer = SentenceTokenizer()
        else:
            self.tokenizer = tokenizer

    def get_text_overlaps(self, text1, text2, min_length=20, tokenize=True):
        """
        :param text1: A piece of text
        :param text2: Another piece of text to compare against the first
        :param min_length: Minimum length of overlapping text to identify
        :param tokenize: Whether or not to tokenize the results; \
        if False, a single block of concatenated text will be returned (default = True)
        :return:
        """

        if tokenize:
            valid_tokens = [t.strip() for t in self.tokenizer.tokenize(". ".join([text1, text2]))]
        fragments = []
        s = SequenceMatcher(None, text1, text2, autojunk=True)
        for block in s.get_matching_blocks():
            if block.size >= min_length:
                for token in self.tokenizer.tokenize(text1[block.a:(block.a + block.size)], min_length=min_length):
                    if len(token) >= min_length:
                        token = token.strip()
                        if not tokenize or token in valid_tokens:
                            fragments.append(token)

        return fragments

    def get_largest_overlap(self, text1, text2):

        s = SequenceMatcher(None, text1, text2)
        pos_a, pos_b, size = s.find_longest_match(0, len(text1), 0, len(text2))
        return text1[pos_a:pos_a + size]


class TextCleaner(object):
    """
    A class for cleaning text up, in preparation for NLP, etc.  Attempts to decode the text.  Then lowercases,
    expands contractions, removes punctuation, lemmatizes, removes stopwords and words less than three characters,
    and consolidates whitespace.

    :param lemmatize: Whether or not to lemmatize the tokens (default = True)
    :param tokenizer: Tokenizer to use (default = nltk.WhitespaceTokenizer())
    :param replacers: A list of tuples, each with a regex pattern followed by the string/pattern to replace them with.\
    Anything passed here will be used in addition to a set of built-in replacement patterns for common contractions.
    :param lemmatizer: Lemmatizer to use (default = nltk.WordNetLemmatizer())
    :param stopwords: The set of stopwords to remove (default = nltk.corpus.stopwords.words('english'))
    :param lowercase: Whether or not to lowercase the string (default = True)
    :param remove_urls: Whether or not to remove URLs and links from the text (default = True)
    :param throw_loud_fail: bool; does not remove all remove nonascii characters if false
    :param strip_html: Whether or not to remove contents wrapped in HTML tags (default = False)
    :param filter_pos: A list of WordNet parts-of-speech tags to keep; \
    if provided, all other words will be removed (default = None)
    """

    def __init__(
        self,
        throw_loud_fail=False,
        filter_pos=None,
        lemmatize=True,
        lemmatizer=None,
        lowercase=True,
        remove_urls=True,
        replacers=None,
        stopwords=None,
        strip_html=False,
        tokenizer=None,
    ):

        self.tokenizer = tokenizer if tokenizer else nltk.WhitespaceTokenizer()
        self.replacers = replacers if replacers else []
        self.replacers.extend([
            (r'won\'t', 'will_not'),
            (r'can\'t', 'cannot'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1>_not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would'),
            (r'it\'s', 'it is')
        ])
        self.replacers = [(re.compile(r"\b{}\b".format(regex[0])), regex[1]) for regex in self.replacers]
        if lemmatize:
            self.lemmatizer = lemmatizer if lemmatizer else nltk.WordNetLemmatizer()
        else:
            self.lemmatizer = None
        if not stopwords:
            stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stopword_regex = re.compile(r"\b({})\b".format(
                r"|".join([re.escape(s) for s in stopwords if s])
            ), re.IGNORECASE)
        if remove_urls:
            self.url_regex = URL_REGEX
#            self.url_regex = re.compile(
#                r"((https?:\/\/(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{2,4000}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))"
#            )
        else:
            self.url_regex = None

        self.filter_pos = filter_pos
        self.lowercase = lowercase
        self.throw_loud_fail = throw_loud_fail
        self.strip_html = strip_html
        self.final_regex = re.compile(r'\w*\d\w*')

    def clean(self, text):
        """
        :param string: The string to clean
        :return: The cleaned string
        """

        # try to remove any html tags in the string
        if self.strip_html:
            text = strip_html(text)

        # try to encode everything as utf-8
        text = _decode_text(text, self.throw_loud_fail)

        if self.filter_pos:
            text = filter_parts_of_speech(text, self.filter_pos)
        if self.lowercase:
            text = str(text).lower()
        if self.url_regex:
            text = self.url_regex.sub(' ', text)

        for regex, replace in self.replacers:
            text = regex.sub(replace, text) # expand contractions
        text = self.stopword_regex.sub('', text)
        text = re.sub(r'\W+', ' ', text)  # remove punctuation
        text = self.tokenizer.tokenize(text)  # split on whitespace

        if self.lemmatizer:
            text = [self.lemmatizer.lemmatize(word) for word in text]

        text = " ".join([word for word in text if len(word) > 2])
        if self.lemmatizer:
            text = self.stopword_regex.sub('', text)
            text = re.sub(r'\W+', ' ', text)

        text = self.final_regex.sub("", text)
        return text


class TextDataFrame(object):
    """
    :param df: A dataframe of documents.  Must contain a column with text.
    :param text_column: The name of the column in the dataframe that contains the text
    :param min_frequency: The minimum number of documents a word must be found in for it to be used in \
        document comparisons (default = 1)
    :param vectorizer_kwargs: All remaining keyword arguments are passed to TfidfVectorizer
    """

    def __init__(self, df, text_column, min_frequency=1, **vectorizer_kwargs):

        self.corpus = df
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(
            min_df=min_frequency,
            decode_error='ignore',
            **vectorizer_kwargs
        )
        self.tfidf = self.vectorizer.fit_transform(df[text_column])


    def search_corpus(self, text):
        """
        Compares the provided text against the documents in the corpus and returns the most similar documents. \
        A new column called 'cosine_similarity' is generated, which is used to sort and return the dataframe.

        :param text: The text to compare documents against
        :return: The corpus dataframe sorted by cosine similarity
        """

        similarities = cosine_similarity(
            self.vectorizer.transform([text]),
            self.tfidf
        )
        self.corpus['search_cosine_similarity'] = similarities[0]
        return self.corpus.sort_values("search_cosine_similarity", ascending=False)

    def match_text_to_corpus(self, match_list, allow_multiple=False, min_similarity=.9):
        """

        :param match_list: A list of strings (other documents) to be matched to documents in the dataframe
        :param allow_multiple: If set to True and your corpus contains duplicates, they will all be matched to \
        their best match in match_list.  If False (default), only the first row will be matched.
        :param min_similarity: Minimum cosine similarity required for any match to be made.
        :return: Your corpus dataframe, with new columns match_text, match_index, and cosine_similarity
        """
        similarities = cosine_similarity(
            self.tfidf,
            self.vectorizer.transform(match_list)
        )
        self.corpus["match_text"] = None
        self.corpus["match_index"] = None
        self.corpus["cosine_similarity"] = None

        for index, row in tqdm(self.corpus.iterrows(), desc="Matching items to corpus"):
            row = self.corpus.iloc[index]
            if is_null(row["match_index"]):
                for i, sim in [s for s in
                               sorted(zip(list(range(0, len(match_list))), similarities[self.corpus.index.get_loc(index)]),
                                      key=lambda x: x[1], reverse=True) if s[1] >= min_similarity]:
                    if i not in self.corpus["match_index"].unique():
                        if allow_multiple:
                            self.corpus.loc[self.corpus[self.text_column] == row[self.text_column], 'match_index'] = i
                            self.corpus.loc[self.corpus[self.text_column] == row[self.text_column], 'match_text'] = \
                            match_list[i]
                            self.corpus.loc[self.corpus[self.text_column] == row[self.text_column], "cosine_similarity"] = sim
                        else:
                            self.corpus.loc[index, 'match_index'] = i
                            self.corpus.loc[index, 'match_text'] = match_list[i]
                            self.corpus.loc[index, "cosine_similarity"] = sim
                        break

        return self.corpus


    def extract_corpus_fragments(self, scan_top_n_matches_per_doc=20, min_fragment_length=15):
        """
        Iterate over the corpus dataframe and, for each document, scan the most similar other documents in the corpus. \
        During each comparison, overlapping fragments are identified.  This can be useful for identifying common \
        boilerplate sentences, repeated paragraphs, etc.

        :param scan_top_n_matches_per_doc: The number of other documents to compare each document against.
        :param min_fragment_length: The minimum character length a fragment must have to be extracted.
        :return: A list of fragments that were found.
        """

        text_overlap_extractor = TextOverlapExtractor()

        similarity_matrix = cosine_similarity(self.tfidf)
        min_similarity = np.average([np.average(row) for row in similarity_matrix])

        combos = []
        for i in range(0, len(self.corpus.index)):
            combos.extend(
                [(i,  cos_similarity[0]) for cos_similarity in
                 sorted(zip(list(range(i + 1, len(self.corpus.index))), similarity_matrix[i][i + 1:]), reverse=True)
                 if  cos_similarity[1] < .997 and  cos_similarity[1] >= min_similarity
                 ][:scan_top_n_matches_per_doc]
            )
        fragments = []
        for i,  cos_similarity in tqdm(combos, desc="Extracting fragments"):
            for frag in text_overlap_extractor.get_text_overlaps(
                    self.corpus.iloc[i][self.text_column],
                    self.corpus.iloc[cos_similarity][self.text_column],
                    min_length=min_fragment_length
            ):
                if frag not in fragments:
                    fragments.append(frag)

        return fragments

    def find_duplicates(self, tfidf_threshold=.9, fuzzy_ratio_threshold=90, allow_partial=False,
                        max_partial_difference=40, filter_function=None, partial_ratio_timeout=5, decode_text=False):

        """
        Search for duplicates by using cosine similarity and Levenshtein ratios.  This will struggle with large \
        corpora, so we recommend trying to filter down to potential duplicates first.  The corpus will first be \
        scanned for document pairs with a cosine similarity greater or equal to the `tfidf_threshold`.  Then, \
        each of these pairs will be compared using the more stringent `fuzzy_ratio_threshold`.

        :param tfidf_threshold: Minimum cosine similarity for two documents to be considered potential dupes.
        :param fuzzy_ratio_threshold: The required Levenshtein ratio to consider two documents duplicates.
        :param filter_function: An optional function that allows for more complex filtering.  The function must accept \
        the following parameters: text1, text2, cosine_similarity, fuzzy_ratio.  Must return True or False, indicating \
        whether the two documents should be considered duplicates.
        :return: A list of lists, containing groups of duplicate documents \
        (represented as rows from the corpus dataframe)

        """

        text = self.corpus[self.text_column]
        if decode_text:
            text = text.map(_decode_text)

        groups = {}
        #compute cosine similarity between the inputs in tf.idf matrix
        similarity_matrix = cosine_similarity(self.tfidf)
        threshold_filter_matrix = similarity_matrix >= tfidf_threshold

        #return the  location of the similarity matrix that satisfies the threshold
        similarity_matrix = np.where(threshold_filter_matrix, similarity_matrix, None)

        #create pairs in the similarity matrix
        pairs = np.argwhere(similarity_matrix)
        pairs = sorted(pairs, key=lambda x: similarity_matrix[x[0]][x[1]], reverse=True)
        pairs = [p for p in pairs if p[0] > p[1]]

        for i, j in tqdm(pairs, desc="Scanning pairs"):
            sim = similarity_matrix[i][j]
            ratio = get_fuzzy_ratio(text.iloc[i], text.iloc[j])
            if ratio < fuzzy_ratio_threshold and allow_partial:
                try:
                    partial_ratio = get_fuzzy_partial_ratio(text.iloc[i], text.iloc[j],
                                                            timeout=partial_ratio_timeout)
                except (MemoryError, TimeoutException):
                    partial_ratio = None
                except Exception as e:
                    print(e)
                    partial_ratio = None
                if partial_ratio and abs(ratio - partial_ratio) <= max_partial_difference:
                    ratio = max([ratio, partial_ratio])
            if ratio >= fuzzy_ratio_threshold and (
                not filter_function or filter_function(self.corpus.iloc[i], self.corpus.iloc[j], sim, ratio)):
                if i not in list(groups.keys()) and j not in list(groups.keys()):
                    new_group = set([i, j])
                    groups[i] = new_group
                    groups[j] = new_group
                elif i in list(groups.keys()) and j not in list(groups.keys()):
                    groups[j] = groups[i]
                elif j in list(groups.keys()) and i not in list(groups.keys()):
                    groups[i] = groups[j]
                else:
                    groups[i].add(j)
                    groups[j].add(i)

        duplicates = []
        final_groups = []
        for g in groups.values():
            if g not in final_groups:
                final_groups.append(g)
                duplicates.append(self.corpus.iloc[list(g)])

        return duplicates

    def mutual_info(
            self,
            outcome_col,
            weight_col=None,
            sample_size=None,
            sort_by="MI",
            top_n=40,
            l=0,
            normalize=True,
            return_raw=False
    ):
        """
        :param outcome_col: The name of the column with the binary outcome variable
        :param top_n: The number of features for each partition you want to return
        :param l: An optional Laplace smoothing parameter
        :param normalize: Toggle normalization on or off (to control for feature prevalance), on by default
        :param return_raw: Return the raw mutual info dataframe instead of sorting and splitting the results
        :return:

        """

        if sample_size:
            df = self.corpus.sample(n=sample_size).reset_index()
        else:
            df = self.corpus
        if weight_col:
            df = df.dropna(subset=[self.text_column, outcome_col, weight_col]).reset_index()
        else:
            df = df.dropna(subset=[self.text_column, outcome_col]).reset_index()
        y = df[outcome_col]
        x = self.vectorizer.fit_transform(df[self.text_column])
        weights = None
        if weight_col:
            weights = df[weight_col]

        return compute_mutual_info(
            y,
            x,
            weights=weights,
            col_names=self.vectorizer.get_feature_names(),
            l=l,
            normalize=normalize,
            top_n=top_n,
            sort_by=sort_by,
            return_raw=return_raw
        )

    def kmeans_clusters(self, k=10):

        self.corpus["kmeans"] = compute_kmeans_clusters(self.tfidf, k=k, return_score=False)
        print("KMeans clusters saved to self.corpus['kmeans']")

    def hdbscan_clusters(self, min_cluster_size=100, min_samples=1):

        self.corpus["hdbscan"] = compute_hdbscan_clusters(self.tfidf, min_cluster_size=min_cluster_size, min_samples=min_samples)
        print("HDBSCAN clusters saved to self.corpus['hdbscan']")

    def top_cluster_terms(self, cluster_col, min_size=50, top_n=10):

        dummies = pd.get_dummies(self.corpus[cluster_col], prefix=cluster_col)
        cluster_df = pd.concat([self.corpus, dummies], axis=1)

        terms = {}
        for cluster in cluster_df[cluster_col].unique():
            if is_not_null(cluster) and len(cluster_df[cluster_df[cluster_col] == cluster]) >= min_size:
                self.corpus["{}_{}".format(cluster_col, cluster)] = cluster_df["{}_{}".format(cluster_col, cluster)]
                minfo = self.mutual_info("{}_{}".format(cluster_col, cluster),
                                         top_n=top_n, return_raw=True)
                del self.corpus["{}_{}".format(cluster_col, cluster)]
                minfo = minfo[minfo["MI1"] > 0].sort_values("MI1", ascending=False)[:top_n]
                terms[cluster] = minfo.index.values
                print("Cluster #{}, {} documents: {}".format(
                    cluster,
                    len(cluster_df[cluster_df[cluster_col] == cluster]),
                    minfo.index.values
                ))
        return terms

    def pca_components(self, k=20):

        """
        Saves the PCA components to self.corpus as new columns ('pca_1', 'pca_2', etc.),
        saves the top component for each document as self.corpus['pca'], and returns
        the features-component matrix

        :param k: Number of dimensions to extract
        :return: A dataframe of (features x components)
        """

        components, documents = get_pca(self.tfidf, feature_names=self.vectorizer.get_feature_names(), k=k)
        for col in documents.columns:
            self.corpus[col] = documents[col]
        print("Top PCA dimensions saved as clusters to self.corpus['pca']")
        return components

    def lsa_components(self, k=20):

        """
        Saves the LSA components to self.corpus as new columns ('lsa_1', 'lsa_2', etc.),
        saves the top component for each document as self.corpus['lsa'], and returns
        the features-component matrix

        :param k: Number of dimensions to extract
        :return: A dataframe of (features x components)
        """

        components, documents = get_lsa(self.tfidf, feature_names=self.vectorizer.get_feature_names(), k=k)
        for col in documents.columns:
            self.corpus[col] = documents[col]
        print("Top LSA dimensions saved as clusters to self.corpus['lsa']")
        return components

    def get_top_documents(self, component_prefix='cluster', top_n=5):

        """
        Use after running get_pca_components or get_lsa_components

        :param component_prefix: 'lsa' or 'pca' (you must first run get_pca_components or get_lsa_components)
        :param top_n: Number of documents to return for each component
        :return: A dictionary where keys are the component, and values are the top_n document indices (or text, if text_column is provided) for each component
        """

        top_docs = {}
        for col in [c for c in self.corpus.columns if c.startswith("{}_".format(component_prefix))]:
            docs = self.corpus[self.corpus[component_prefix] == col].sort_values(col, ascending=False)[:top_n]
            top_docs[col] = docs[self.text_column].values
        return top_docs

    def make_word_cooccurrence_matrix(
            self,
            normalize=False,
            min_frequency=10,
            max_frequency=.5
    ):

        """
        Use to produce word co-occurrence matrices. Based on a helpful StackOverflow post:
        https://stackoverflow.com/questions/35562789/how-do-i-calculate-a-word-word-co-occurrence-matrix-with-sklearn
        :param normalize: If True, will be normalized
        :param min_frequency: The minimum document frequency required for a term to be included
        :param max_frequency: The maximum proportion of documents containing a term allowed to include the term
        :return:
        """

        text = self.corpus[self.text_column]
        cv = CountVectorizer(
            ngram_range=(1,1),
            stop_words = 'english',
            min_df=min_frequency,
            max_df=max_frequency
        )
        mat = cv.fit_transform(text)
        mat[mat > 0] = 1  # this makes sure that we're counting number of documents words have in common \
        # and not weighting by the frequency of one of the words in a single document, which can lead to spurious links
        names = cv.get_feature_names()
        mat = (mat.T * mat)  # compute the document-document matrix
        if normalize:
            diag = sp.diags(1.0 / mat.diagonal())
            mat = diag * mat
        mat.setdiag(0)
        matrix = pd.DataFrame(data=mat.todense(), columns=names, index=names)

        return matrix

    def make_document_cooccurrence_matrix(self, normalize=False):

        """
        Use to produce document co-occurrence matrices. Based on a helpful StackOverflow post:
        https://stackoverflow.com/questions/35562789/how-do-i-calculate-a-word-word-co-occurrence-matrix-with-sklearn
        :param normalize: If True, will be normalized
        :return:
        """

        text = self.corpus[self.text_column]
        cv = CountVectorizer(ngram_range=(1, 1), stop_words='english')
        mat = cv.fit_transform(text)
        mat[mat > 0] = 1  # this makes sure that we're counting number of words documents have in common \
        # and not weighting by the frequency of one of the words in a single document, which can lead to spurious links
        names = text.index
        mat = (mat * mat.T)  # compute the word-word matrix
        if normalize:
            diag = sp.diags(1.0 / mat.diagonal())
            mat = diag * mat
        mat.setdiag(0)
        matrix = pd.DataFrame(data=mat.todense(), columns=names, index=names)

        return matrix
