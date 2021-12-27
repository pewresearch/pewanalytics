from __future__ import absolute_import

import re
import copy
import pandas as pd
import numpy as np
import scipy.sparse as sp

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import wordnet

from tqdm import tqdm
from stopit import ThreadingTimeout as Timeout

try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from stopit import TimeoutException

from pewtils import is_null, is_not_null
from pewtils.http import strip_html
from pewtils import decode_text as _decode_text
from pewtils.regex import URL_REGEX

from pewanalytics.stats.clustering import (
    compute_hdbscan_clusters,
    compute_kmeans_clusters,
)
from pewanalytics.stats.mutual_info import compute_mutual_info
from pewanalytics.stats.dimensionality_reduction import get_lsa, get_pca


def has_fragment(text, fragment):

    """
    Checks whether a substring ("fragment") is contained within a larger string ("text"). Uses the \
    :py:func:`pewtils.decode_text` function to process both the text and the fragment when running this check.

    :param text: The text to search
    :type text: str
    :param fragment: The fragment to search for
    :type fragment: str
    :return: Whether or not the text contains the fragment
    :rtype: bool

    Usage::

        from pewanalytics.text import has_fragment

        text = "testing one two three"

        >>> has_fragment(text, "one two")
        True

        >>> has_fragment(text, "four")
        False
    """

    return any([(fragment in text), (_decode_text(fragment) in _decode_text(text))])


def remove_fragments(text, fragments, throw_loud_fail=False):

    """
    Iteratively remove fragments from a string.

    :param text: The text toremove the fragments from
    :type text: str
    :param fragments: A list of string fragments to search for and remove
    :type fragments: list
    :param throw_loud_fail: bool; whether or not to raise an error if text decoding fails (default=False)
    :type throw_loud_fail: bool
    :return: The original string, minus any parts that matched the fragments provided
    :rtype: str

    Usage::

        from pewanalytics.text import remove_fragments

        text = "testing one two three"

        >>> remove_fragments(text, ["one two"])
        "testing  three"

        >>> remove_fragments(text, ["testing", "three"])
        " one two "
    """

    for f in fragments:
        new_text = text.replace(f, "")
        # if the new text is the same as previous, try decoding
        if new_text == text:
            new_text = _decode_text(text, throw_loud_fail).replace(
                _decode_text(f, throw_loud_fail), ""
            )
        # if the new text is still the same as previous, then new text is None
        if new_text == text:
            new_text = None
        if new_text:
            text = new_text
    return text


def filter_parts_of_speech(text, filter_pos=None, exclude=False):

    """
    Retain words associated with parts of speech in the text if ``exclude=False``.
    If ``exclude=True``, exclude words associated with parts of speech.
    Default is Noun (NN), Proper Noun (NNP) and Adjective (JJ)

    | The full list of POS is here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    :param text: The string to process
    :type text: str
    :param filter_pos: Array of part of speech tags (default is 'NN', 'NNP', and 'JJ')
    :type filter_pos: list
    :param exclude: If ``True``, the function will remove words that match to the specified parts of speech; by default \
    this function *filters to* POS matches instead.
    :return: A string comprised solely of words that matched (or did not match) to the specified parts of speech, \
    depending on the value of ``exclude``
    :rtype: str

    Usage::

        from pewanalytics.text import filter_parts_of_speech

        text = "This is a very exciting sentence that can serve as a functional example"

        >>> filter_parts_of_speech(text, filter_pos=["NN"])
        'sentence example'

        >>> filter_parts_of_speech(text, filter_pos=["JJ"], exclude=True)
        'This is a very sentence that can serve as a example'

    """

    if not filter_pos:
        filter_pos = ("NN", "NNP", "JJ")
    text = text.split()
    tagged_words = nltk.pos_tag(text)
    if not exclude:
        valid = [word[0] for word in tagged_words if word[1] in filter_pos]
    else:
        valid = [word[0] for word in tagged_words if word[1] not in filter_pos]
    return " ".join(valid)


def get_fuzzy_ratio(text1, text2, throw_loud_fail=False):

    """
    Uses Levenshtein Distance to calculate similarity of two strings.  Measures how the edit distance compares
    to the overall length of the texts. Uses the :py:mod:`fuzzywuzzy` library in Python 2, and the :py:mod:`rapidfuzz` \
    library in Python 3.

    :param text1: First string
    :type text1: str
    :param text2: Second string
    :type text1: str
    :param throw_loud_fail: bool; whether or not to raise an error if text decoding fails (default=False)
    :type throw_loud_fail: bool
    :return: The Levenshtein ratio between the two strings
    :rtype: float

    Usage::

        from pewanalytics.text import get_fuzzy_ratio

        text1 = "This is a sentence."
        text2 = "This is a slightly difference sentence."

        >>> get_fuzzy_ratio(text1, text2)
        64.28571428571428

    """

    try:
        return fuzz.ratio(text1, text2)
    except (UnicodeDecodeError, UnicodeEncodeError):
        return fuzz.ratio(
            _decode_text(text1, throw_loud_fail), _decode_text(text2, throw_loud_fail)
        )


def get_fuzzy_partial_ratio(text1, text2, throw_loud_fail=False, timeout=5):

    """
    Useful to calculate similarity of two strings that are of noticeably different lengths.  Allows for the possibility
    that one text is a subset of the other; finds the largest overlap and computes the Levenshtein ratio on that.

    :param text1: First string
    :type text1: str
    :param text2: Second string
    :type text2: str
    :param timeout: The number of seconds to wait before giving up
    :type timeout: int
    :param throw_loud_fail: bool; whether or not to raise an error if text decoding fails (default=False)
    :type throw_loud_fail: bool
    :return: The partial Levenshtein ratio between the two texts
    :rtype: float

    :accepts kwarg timeout:

    Usage::

        from pewanalytics.text import get_partial_fuzzy_ratio

        text1 = "This is a sentence."
        text2 = "This is a sentence, but with more text."

        >>> get_partial_fuzzy_ratio(text1, text2)
        100.0

    """

    partial_ratio = None
    with Timeout(timeout, swallow_exc=True):
        try:
            partial_ratio = fuzz.partial_ratio(text1, text2)
        except (UnicodeDecodeError, UnicodeEncodeError):
            partial_ratio = fuzz.partial_ratio(
                _decode_text(text1, throw_loud_fail),
                _decode_text(text2, throw_loud_fail),
            )
        return partial_ratio


class SentenceTokenizer(object):

    """
    Initializes a tokenizer that can be be used to break text into tokens using the ``tokenize`` function

    :param base_tokenizer: The tokenizer to use (default = NLTK's English Punkt tokenizer)
    :param regex_split_trailing: A compiled regex object used to define the end of sentences
    :param regex_split_leading: A compiled regex object used to define the beginning of sentences

    Usage::

        from pewanalytics.text import SentenceTokenizer
        import re

        text = "This is a sentence. This is another sentence - and maybe a third sentence. And yet a fourth sentence."

        >>> tokenizer = SentenceTokenizer()
        >>> tokenizer.tokenize(text)
        ['This is a sentence.',
         'This is another sentence - and maybe a third sentence.',
         'And yet a fourth sentence.']

        >>> tokenizer = SentenceTokenizer(regex_split_leading=re.compile(r"\-"))
        >>> tokenizer.tokenize(text)
        ['This is a sentence.',
         'This is another sentence',
         'and maybe a third sentence.',
         'And yet a fourth sentence.']

    """

    def __init__(
        self, base_tokenizer=None, regex_split_trailing=None, regex_split_leading=None
    ):
        self.base_tokenizer = (
            base_tokenizer
            if base_tokenizer
            else nltk.data.load("tokenizers/punkt/english.pickle")
        )
        self.regex_split_trailing = regex_split_trailing
        self.regex_split_leading = regex_split_leading

    def tokenize(self, text, throw_loud_fail=False, min_length=None):

        """
        Tokenizes the text.

        :param text: The text to tokenize
        :type text: str
        :param throw_loud_fail: Whether or not to raise an error if text decoding fails (default=False)
        :type throw_loud_fail: bool
        :param min_length: The minimum acceptable length of a sentence (if a token is shorter than this, it will be \
        considered part of the preceding sentence) (default=None)
        :type min_length: int
        :return: A list of sentences
        :rtype: list
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
                    if subt_lead != "":
                        token_group.append(subt_lead)
                    if len(leaders) == 0 or subt_lead not in leaders:
                        partial_tokens.append("".join(token_group))
                        token_group = []
                if len(token_group) > 0:
                    partial_tokens.append("".join([t for t in token_group if t != ""]))
        if len(token_group) > 0:
            partial_tokens.append("".join([t for t in token_group if t != ""]))

        if not self.regex_split_trailing:
            final_tokens = partial_tokens
        else:
            final_tokens = []
            token_group = []
            for t in partial_tokens:
                trailers = self.regex_split_trailing.findall(t)
                token_group = []
                for subt_trail in self.regex_split_trailing.split(t):
                    if subt_trail != "":
                        token_group.append(subt_trail)
                    if len(trailers) == 0 or subt_trail in trailers:
                        final_tokens.append("".join(token_group))
                        token_group = []
                if len(token_group) > 0:
                    final_tokens.append("".join([t for t in token_group if t != ""]))
            if len(token_group) > 0:
                final_tokens.append("".join([t for t in token_group if t != ""]))

        final_tokens = [t.strip() for t in final_tokens]

        if min_length:
            final_tokens = [f for f in final_tokens if len(f) >= min_length]

        return final_tokens


class TextOverlapExtractor(object):

    """
    A helper class designed to identify overlapping sections between two strings.

    :param tokenizer: The tokenizer to use (default = SentenceTokenizer())
    """

    def __init__(self, tokenizer=None):

        if not tokenizer:
            self.tokenizer = SentenceTokenizer()
        else:
            self.tokenizer = tokenizer

    def get_text_overlaps(self, text1, text2, min_length=20, tokenize=True):

        """
        Extracts all overlapping segments of at least ``min_length`` characters between the two texts. If ``tokenize=True``
        then only tokens that appear fully in both texts will be extracted. For example:

        :param text1: A piece of text
        :type text1: str
        :param text2: Another piece of text to compare against the first
        :type text2: str
        :param min_length: The minimum size of the overlap to be considered (number of characters)
        :type min_length: int
        :param tokenize: If True, overlapping segments will only be included if they consist of atomic tokens; \
        overlaps that consist of only part of a token will be excluded. By default, the text is tokenize into \
        sentences based on punctuation. (default=True)
        :type tokenize: bool
        :return: A list of all of the identified overlapping segments
        :rtype: list

        Usage::

            from pewanalytics.text import TextOverlapExtractor

            text1 = "This is a sentence. This is another sentence. And a third sentence. And yet a fourth sentence."
            text2 = "This is a different sentence. This is another sentence. And a third sentence. But the fourth \
            sentence is different too."

            >>> extractor = TextOverlapExtractor()

            >>> extractor.get_text_overlaps(text1, text2, min_length=10, tokenize=False)
            [' sentence. This is another sentence. And a third sentence. ', ' fourth sentence']

            >>> extractor.get_text_overlaps(text1, text2, min_length=10, tokenize=True)
            ['This is another sentence.', 'And a third sentence.']

        """

        valid_tokens = None
        if tokenize:
            valid_tokens = [
                t.strip() for t in self.tokenizer.tokenize(". ".join([text1, text2]))
            ]
        fragments = []
        s = SequenceMatcher(None, text1, text2, autojunk=True)
        for block in s.get_matching_blocks():
            if block.size >= min_length:
                overlap = text1[block.a : (block.a + block.size)]
                if tokenize:
                    for token in self.tokenizer.tokenize(
                        overlap, min_length=min_length
                    ):
                        token = token.strip()
                        if not valid_tokens or token in valid_tokens:
                            fragments.append(token)
                elif len(overlap) >= min_length:
                    fragments.append(overlap)

        return fragments

    def get_largest_overlap(self, text1, text2):

        """
        Returns the largest overlapping segment of text between the two texts (this doesn't use the tokenizer).

        :param text1: A piece of text
        :type text1: str
        :param text2: Another piece of text to compare against the first
        :type text2: str
        :return: The largest substring that occurs in both texts
        :rtype: str

        Usage::

            from pewanalytics.text import TextOverlapExtractor

            text1 = "Overlaping section, unique text another overlapping section"
            text2 = "Overlapping section, another overlapping section"


            >>> extractor = TextOverlapExtractor()

            >>> extractor.get_largest_overlap(text1, text2)
            ' another overlapping section'

        """

        s = SequenceMatcher(None, text1, text2)
        pos_a, pos_b, size = s.find_longest_match(0, len(text1), 0, len(text2))
        return text1[pos_a : pos_a + size]


class TextCleaner(object):

    """
    A class for cleaning text up, in preparation for NLP, etc.  Attempts to decode the text.

    This function performs for the following cleaning tasks, in sequence:

        - Removes HTML tags (optional)
        - Decodes the text
        - Filters out specified parts of speech (optional)
        - Converts text to lowercase (optional)
        - Removes URLs (optional)
        - Expands contractions
        - Removes stopwords
        - Lemmatizes or stems (optional)
        - Removes words less than three characters
        - Removes punctuation
        - Consolidates whitespace

    :param process_method: Options are "lemmatize", "stem", or None (default = "lemmatize")
    :type process_method: str
    :param processor: A lemmatizer or stemmer with a "lemmatize" or "stem" function (default for \
    process_method="lemmatize" is nltk.WordNetLemmatizer(); default for process_method="stem" is nltk.SnowballStemmer())
    :param filter_pos: A list of WordNet parts-of-speech tags to keep; \
    if provided, all other words will be removed (default = None)
    :type filter_pos: list
    :param lowercase: Whether or not to lowercase the string (default = True)
    :type lowercase: bool
    :param remove_urls: Whether or not to remove URLs and links from the text (default = True)
    :type remove_urls: bool
    :param replacers: A list of tuples, each with a regex pattern followed by the string/pattern to replace them with. \
    Anything passed here will be used in addition to a set of built-in replacement patterns for common contractions.
    :param stopwords: The set of stopwords to remove (default = nltk.corpus.stopwords.words('english') combined with \
    sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS). If an empty list is passed, no stopwords will be used.
    :type stopwords: set
    :param strip_html: Whether or not to remove contents wrapped in HTML tags (default = False)
    :type strip_html: bool
    :param tokenizer: Tokenizer to use (default = nltk.WhitespaceTokenizer())
    :type replacers: list
    :param throw_loud_fail: bool; whether or not to raise an error if text decoding fails (default=False)
    :type throw_loud_fail: bool

    Usage::

        from pewanalytics.text import TextCleaner

        text = "<body> \
            Here's some example text.</br>It isn't a great example, but it'll do. \
            Of course, there are plenty of other examples we could use though. \
            http://example.com \
            </body>"

        >>> cleaner = TextCleaner(process_method="stem")
        >>> cleaner.clean(text)
        'exampl is_not great exampl cours plenti exampl could use though'

        >>> cleaner = TextCleaner(process_method="stem", stopwords=["my_custom_stopword"], strip_html=True)
        >>> cleaner.clean(text)
        'here some exampl is_not great exampl but will cours there are plenti other exampl could use though'

        >>> cleaner = TextCleaner(process_method="lemmatize", strip_html=True)
        >>> cleaner.clean(text)
        'example is_not great example course plenty example could use though'

        >>> cleaner = TextCleaner(process_method="lemmatize", remove_urls=False, strip_html=True)
        >>> cleaner.clean(text)
        'example text is_not great example course plenty example could use though http example com'

        >>> cleaner = TextCleaner(process_method="stem", strip_html=False)
        >>> cleaner.clean(text)
        'example text is_not great example course plenty example could use though http example com'

        >>> cleaner = TextCleaner(process_method="stem", filter_pos=["JJ"], strip_html=True)
        >>> cleaner.clean(text)
        'great though'

    """

    def __init__(
        self,
        process_method="lemmatize",
        processor=None,
        filter_pos=None,
        lowercase=True,
        remove_urls=True,
        replacers=None,
        stopwords=None,
        strip_html=False,
        tokenizer=None,
        throw_loud_fail=False,
    ):

        self.tokenizer = tokenizer if tokenizer else nltk.WhitespaceTokenizer()
        self.replacers = replacers if replacers else []
        self.replacers.extend(
            [
                (r"won\'t", "will_not"),
                (r"can\'t", "cannot"),
                (r"i\'m", "i am"),
                (r"ain\'t", "is not"),
                (r"(\w+)\'ll", r"\g<1> will"),
                (r"(\w+)n\'t", r"\g<1>_not"),
                (r"(\w+)\'ve", r"\g<1> have"),
                (r"(\w+)\'re", r"\g<1> are"),
                (r"(\w+)\'d", r"\g<1> would"),
                (r"it\'s", "it is"),
            ]
        )  # Borrowed from NLTK cookbook
        self.replacers = [
            (re.compile(r"\b{}\b".format(regex[0])), regex[1])
            for regex in self.replacers
        ]
        if process_method == "lemmatize":
            self.processor = processor if processor else nltk.WordNetLemmatizer()
            self.process_func = self.processor.lemmatize
        elif process_method == "stem":
            self.processor = processor if processor else nltk.SnowballStemmer("english")
            self.process_func = self.processor.stem
        else:
            self.processor = None
            self.process_func = None
        if is_null(stopwords):
            stopwords = set.union(
                set(nltk.corpus.stopwords.words("english")), set(ENGLISH_STOP_WORDS)
            )
        self.stopword_regex = re.compile(
            r"\b({})\b".format(r"|".join([re.escape(s) for s in stopwords if s])),
            re.IGNORECASE,
        )
        if remove_urls:
            self.url_regex = URL_REGEX
        else:
            self.url_regex = None

        self.filter_pos = filter_pos
        self.lowercase = lowercase
        self.throw_loud_fail = throw_loud_fail
        self.strip_html = strip_html
        self.final_regex = re.compile(r"\w*\d\w*")

    def clean(self, text):
        """
        Cleans the text.

        :param text: The string to clean
        :type text: str
        :return: The cleaned string
        :rtype: str
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
            text = self.url_regex.sub(" ", text)

        for regex, replace in self.replacers:
            text = regex.sub(replace, text)  # expand contractions
        text = self.stopword_regex.sub("", text)
        text = re.sub(r"\W+", " ", text)  # remove punctuation
        text = self.tokenizer.tokenize(text)  # split on whitespace

        if self.processor:
            text = [self.process_func(word) for word in text]

        text = " ".join([word for word in text if len(word) > 2])
        if self.processor:
            text = self.stopword_regex.sub("", text)
            text = re.sub(r"\W+", " ", text)

        text = self.final_regex.sub("", text)
        return text


class TextDataFrame(object):

    """
    This is a class full of functions for working with dataframes of documents. It contains utilities for identifying \
    potential duplicates, identifying recurring segments of text, computing metrics like mutual information, \
    extracting clusters of documents, and more.

    Given a :py:class:`pandas.DataFrame` and the name of the column that contains the text to be analyzed, the \
    TextDataFrame will automatically produce a TF-IDF sparse matrix representation of the text upon initialization. \
    All other parameters are passed along to the scikit-learn TfidfVectorizer.

    .. tip:: For more info on the parameters it excepts, refer to the official scikit-learn `TfidfVectorizer \
    documentation \
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_.

    :param df: A :py:class:`pandas.DataFrame` of documents.  Must contain a column with text.
    :param text_column: The name of the column in the :py:class:`pandas.DataFrame` that contains the text
    :type text_column: str
    :param vectorizer_kwargs: All remaining keyword arguments are passed to TfidfVectorizer

    Usage::

        from pewanalytics.text import TextDataFrame
        import pandas as pd
        import nltk

        nltk.download("inaugural")
        df = pd.DataFrame([
            {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
        ])
        # Let's remove new line characters so we can print the output in the docstrings
        df['text'] = df['text'].str.replace("\\n", " ")

        # And now let's create some additional variables to group our data
        df['year'] = df['speech'].map(lambda x: int(x.split("-")[0]))
        df['21st_century'] = df['year'].map(lambda x: 1 if x >= 2000 else 0)

        # And we'll also create some artificial duplicates in the dataset
        df = df.append(df.tail(2)).reset_index()

        >>> tdf = TextDataFrame(df, "text", stop_words="english", ngram_range=(1, 2))
        >>> tdf_dense = pd.DataFrame(tdf.tfidf.todense(), columns=tdf.vectorizer.get_feature_names()).head(5)
        >>> tdf_dense.loc[:, (tdf_dense != 0).any(axis=0)]
        	    14th	14th day	 abandon  abandon government... zeal inspires	zeal purity	zeal rely	zeal wisdom
        0	0.034014	0.034014	0.000000	       0.000000	...      0.000000	   0.000000	 0.000000	   0.000000
        1	0.000000	0.000000	0.000000	       0.000000	...      0.000000	   0.000000	 0.000000	   0.000000
        2	0.000000	0.000000	0.000000	       0.000000	...      0.000000	   0.000000	 0.000000	   0.000000
        3	0.000000	0.000000	0.020984	       0.030686	...      0.000000	   0.000000	 0.030686	   0.000000
        4	0.000000	0.000000	0.000000	       0.000000	...      0.026539	   0.026539	 0.000000	   0.026539

    """

    def __init__(self, df, text_column, **vectorizer_kwargs):
        self.corpus = df
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(decode_error="ignore", **vectorizer_kwargs)
        self.tfidf = self.vectorizer.fit_transform(df[text_column])

    def search_corpus(self, text):

        """
        Compares the provided text against the documents in the corpus and returns the most similar documents. \
        A new column called 'cosine_similarity' is generated, which is used to sort and return the \
        :py:class:`pandas.DataFrame`.

        :param text: The text to compare documents against
        :type text: str
        :return: The corpus :py:class:`pandas.DataFrame` sorted by cosine similarity

        Usage::

            >>> tdf.search_corpus('upright zeal')[:5]
                                                            text	search_cosine_similarity
            4	Proceeding, fellow citizens, to that qualifica...	0.030856
            8	Fellow citizens, I shall not attempt to descri...	0.025041
            9	In compliance with an usage coeval with the ex...	0.024922
            27	Fellow citizens, In obedience to the will of t...	0.021272
            10	Fellow citizens, about to undertake the arduou...	0.014791

        """

        similarities = cosine_similarity(self.vectorizer.transform([text]), self.tfidf)
        corpus = copy.deepcopy(self.corpus[[self.text_column]])
        corpus["search_cosine_similarity"] = similarities[0]
        return corpus.sort_values("search_cosine_similarity", ascending=False)

    def match_text_to_corpus(
        self, match_list, allow_multiple=False, min_similarity=0.9
    ):
        """
        Takes a list of text values and attempts to match them to the documents in the :py:class:`pandas.DataFrame`. \
        Each document will be matched to the value in the list to which it is most similar, based on cosine similarity.

        :param match_list: A list of strings (other documents) to be matched to documents in the \
        :py:class:`pandas.DataFrame`
        :type match_list: str
        :param allow_multiple: If set to True, each document in your corpus will be matched with its closes valid \
        match in the list. If set to False (default), documents in the list will only be matched to their best match \
        in the corpus.
        :type allow_multiple: bool
        :param min_similarity: Minimum cosine similarity required for any match to be made.
        :type min_similarity: float
        :return: Your corpus :py:class:`pandas.DataFrame`, with new columns match_text, match_index, \
        and cosine_similarity

        Usage::

            >>> match_df = tdf.match_text_to_corpus(test_excerpt, min_similarity=0.05)
            >>> match_df.sort_values('cosine_similarity')[:2]
                                                             text	                                       match_text	match_index	cosine_similarity
            48	Senator Hatfield, Mr. Chief Justice, Mr. Presi...	In this present crisis, government is not the ...	1	        0.0699283
            43	Vice President Johnson, Mr. Speaker, Mr. Chief...	And so, my fellow Americans: ask not what your...	0	        0.166681

        """
        similarities = cosine_similarity(
            self.tfidf, self.vectorizer.transform(match_list)
        )

        corpus = copy.deepcopy(self.corpus[[self.text_column]])
        corpus["match_text"] = None
        corpus["match_index"] = None
        corpus["cosine_similarity"] = None

        for index, row in tqdm(corpus.iterrows(), desc="Matching items to corpus"):
            row = corpus.iloc[index]
            if is_null(row["match_index"]):
                for i, sim in [
                    s
                    for s in sorted(
                        zip(
                            list(range(0, len(match_list) + 1)),
                            similarities[corpus.index.get_loc(index)],
                        ),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    if s[1] >= min_similarity
                ]:
                    match = True
                    if (
                        not allow_multiple
                        and i
                        in corpus[~corpus["match_index"].isnull()][
                            "match_index"
                        ].unique()
                    ):
                        current_best = corpus.loc[corpus["match_index"] == i][
                            "cosine_similarity"
                        ].max()
                        if sim >= current_best:
                            corpus.loc[corpus["match_index"] == i, "match_text"] = None
                            corpus.loc[
                                corpus["match_index"] == i, "cosine_similarity"
                            ] = None
                            corpus.loc[corpus["match_index"] == i, "match_index"] = None
                        else:
                            match = False
                    if match:
                        corpus.loc[
                            corpus[self.text_column] == row[self.text_column],
                            "match_index",
                        ] = i
                        corpus.loc[
                            corpus[self.text_column] == row[self.text_column],
                            "match_text",
                        ] = match_list[i]
                        corpus.loc[
                            corpus[self.text_column] == row[self.text_column],
                            "cosine_similarity",
                        ] = sim
                        break

        return corpus

    def extract_corpus_fragments(
        self,
        scan_top_n_matches_per_doc=20,
        min_fragment_length=15,
        tokenize=True,
        tokenizer=None,
    ):
        """
        Iterate over the corpus :py:class:`pandas.DataFrame` and, for each document, scan the most similar other \
        documents in the corpus using TF-IDF cosine similarity. During each comparison, overlapping fragments are \
        identified.  This can be useful for identifying common boilerplate sentences, repeated paragraphs, etc. \
        By default, the text is tokenized into complete sentences (so only complete sentences that recur will be \
        returned), but you can set ``tokenize=False`` to get raw segments of text that occur multiple times.

        :param scan_top_n_matches_per_doc: The number of other documents to compare each document against.
        :type scan_top_n_matches_per_doc: int
        :param min_fragment_length: The minimum character length a fragment must have to be extracted.
        :type min_fragment_length: int
        :param tokenize: If True, overlapping segments will only be included if they consist of atomic tokens; \
        overlaps that consist of only part of a token will be excluded. Uses sentence tokenization by default. \
        (default=True)
        :type tokenize: bool
        :param tokenizer: The tokenizer to use, if tokenizing isn't disabled (default = SentenceTokenizer())
        :type tokenizer: object
        :return: A list of fragments that were found.

        .. note:: This function will skip over duplicates if they exist in your data; it only compares documents
            that have less than .997 cosine similarity.

        Usage::

            >>> tdf.extract_corpus_fragments(scan_top_n_matches_per_doc=20, min_fragment_length=25, tokenize=False)
            ['s. Equal and exact justice ',
             'd by the General Government',
             ' of the American people, ',
             'ent of the United States ',
             ' the office of President of the United States ',
             ' preserve, protect, and defend the Constitution of the United States."  ',
             ' to "preserve, protect, and defend',
             ' of the United States are ',
             'e of my countrymen I am about to ',
             'Vice President, Mr. Chief Justice, ',
             ' 200th anniversary as a nation',
             ', and my fellow citizens: ',
             'e United States of America']
        """

        text_overlap_extractor = TextOverlapExtractor(tokenizer=tokenizer)

        similarity_matrix = cosine_similarity(self.tfidf)
        min_similarity = np.average([np.average(row) for row in similarity_matrix])

        combos = []
        for i in range(0, len(self.corpus.index)):
            combos.extend(
                [
                    (i, cos_similarity[0])
                    for cos_similarity in sorted(
                        zip(
                            list(range(i + 1, len(self.corpus.index))),
                            similarity_matrix[i][i + 1 :],
                        ),
                        reverse=True,
                    )
                    if min_similarity <= cos_similarity[1] < 0.997
                ][:scan_top_n_matches_per_doc]
            )
        fragments = []
        for i, cos_similarity in tqdm(combos, desc="Extracting fragments"):
            for frag in text_overlap_extractor.get_text_overlaps(
                self.corpus.iloc[i][self.text_column],
                self.corpus.iloc[cos_similarity][self.text_column],
                min_length=min_fragment_length,
                tokenize=tokenize,
            ):
                if frag not in fragments:
                    fragments.append(frag)

        return fragments

    def find_duplicates(
        self,
        tfidf_threshold=0.9,
        fuzzy_ratio_threshold=90,
        allow_partial=False,
        max_partial_difference=40,
        filter_function=None,
        partial_ratio_timeout=5,
        decode_text=False,
    ):

        """
        Search for duplicates by using cosine similarity and Levenshtein ratios.  This will struggle with large
        corpora, so we recommend trying to filter down to potential duplicates first.  The corpus will first be
        scanned for document pairs with a cosine similarity greater or equal to the ``tfidf_threshold``.  Then,
        each of these pairs will be compared using the more stringent ``fuzzy_ratio_threshold``.

        :param tfidf_threshold: Minimum cosine similarity for two documents to be considered potential dupes.
        :type tfidf_threshold: float
        :param fuzzy_ratio_threshold: The required Levenshtein ratio to consider two documents duplicates.
        :type fuzzy_ratio_threshold: int
        :param allow_partial: Whether or not to allow a partial ratio (if False, absolute ratios will be used)
        :type allow_partial: bool
        :param max_partial_diff: The maximum partial ratio difference allowed for a potential duplicate pair
        :type max_partial_diff: int
        :param filter_function: An optional function that allows for more complex filtering. The function must accept \
        the following parameters: text1, text2, cosine_similarity, fuzzy_ratio.  Must return True or False, \
        indicating whether the two documents should be considered duplicates.
        :param partial_ratio_timeout: How long, in seconds, that the partial ratio is allowed to compute
        :type partial_ratio_timeout: int
        :param decode_text: Whether to decode the text prior to making comparisons
        :type decode_text: bool
        :return: A list of lists, containing groups of duplicate documents (represented as rows from the corpus \
        :py:class:`pandas.DataFrame`)

        Usage::

            >>> tdf.find_duplicates()
            [           speech                                               text  year
            56  2013-Obama.txt  Thank you. Thank you so much.    Vice Presiden...  2013
            56  2013-Obama.txt  Thank you. Thank you so much.    Vice Presiden...  2013

                21st_century
            56             1
            56             1  ,
                        speech                                               text  year
            57  2017-Trump.txt  Chief Justice Roberts, President Carter, Presi...  2017
            57  2017-Trump.txt  Chief Justice Roberts, President Carter, Presi...  2017

                21st_century
            57             1
            57             1  ]

        """

        text = copy.deepcopy(self.corpus[self.text_column])
        if decode_text:
            text = text.map(_decode_text)

        groups = {}
        # compute cosine similarity between the inputs in tf.idf matrix
        similarity_matrix = cosine_similarity(self.tfidf)
        threshold_filter_matrix = similarity_matrix >= tfidf_threshold

        # return the  location of the similarity matrix that satisfies the threshold
        similarity_matrix = np.where(threshold_filter_matrix, similarity_matrix, None)

        # create pairs in the similarity matrix
        pairs = np.argwhere(similarity_matrix)
        pairs = sorted(pairs, key=lambda x: similarity_matrix[x[0]][x[1]], reverse=True)
        pairs = [p for p in pairs if p[0] > p[1]]

        for i, j in tqdm(pairs, desc="Scanning pairs"):
            sim = similarity_matrix[i][j]
            ratio = get_fuzzy_ratio(text.iloc[i], text.iloc[j])
            if ratio < fuzzy_ratio_threshold and allow_partial:
                try:
                    partial_ratio = get_fuzzy_partial_ratio(
                        text.iloc[i], text.iloc[j], timeout=partial_ratio_timeout
                    )
                except (MemoryError, TimeoutException):
                    partial_ratio = None
                except Exception as e:
                    print(e)
                    partial_ratio = None
                if (
                    partial_ratio
                    and abs(ratio - partial_ratio) <= max_partial_difference
                ):
                    ratio = max([ratio, partial_ratio])
            if ratio >= fuzzy_ratio_threshold and (
                not filter_function
                or filter_function(self.corpus.iloc[i], self.corpus.iloc[j], sim, ratio)
            ):
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

    def find_related_keywords(self, keyword, n=25):

        """
        Given a particular keyword, looks for related terms in the corpus using mutual information.

        :param keyword: The keyword to use
        :type keyword: str
        :param n: Number of related terms to return
        :type n: int
        :return: Terms associated with the keyword
        :rtype: list

        Usage::

            >>> tdf.find_related_keywords("war")[:2]
            ['war', 'peace']

            >>> tdf.find_related_keywords("economy")[:2]
            ['economy', 'expenditures']

        """

        self.corpus["temp"] = (
            self.corpus[self.text_column]
            .str.contains(r"\b{}\b".format(keyword), re.IGNORECASE)
            .astype(int)
        )
        mi = self.mutual_info("temp")
        del self.corpus["temp"]

        return list(mi[mi["MI1"] > 0].sort_values("MI1", ascending=False)[:n].index)

    def mutual_info(
        self, outcome_col, weight_col=None, sample_size=None, l=0, normalize=True
    ):

        """
        A wrapper around :py:func:`pewanalytics.stats.mutual_info.compute_mutual_info`

        :param outcome_col: The name of the column with the binary outcome variable
        :type outcome_col: str
        :param weight_col: (Optional) Name of the column to use in weighting
        :type weight_col: str
        :param sample_size: (Optional) If provided, a random sample of this size will be used instead of the full \
        :py:class:`pandas.DataFrame`
        :type sample_size: int
        :param l: An optional Laplace smoothing parameter
        :type l: float
        :param normalize: Toggle normalization on or off (to control for feature prevalence), on by default
        :type normalize: bool
        :return: A :py:class:`pandas.DataFrame` of ngrams and various metrics about them, including mutual information

        Usage::

            >>> results = tdf.mutual_info('21st_century')
            >>> results.sort_values("MI1", ascending=False).index[:25]
            Index(['journey complete', 'jobs', 'make america', 've', 'obama', 'workers',
                   'xand', 'states america', 'america best', 'debates', 'clinton',
                   'president clinton', 'trillions', 'stops right', 'transferring',
                   'president obama', 'stops', 'protected protected', 'transferring power',
                   'nation capital', 'american workers', 'politicians', 'people believe',
                   'borders', 'victories'],
                   dtype='object')

        """

        keep_columns = [self.text_column, outcome_col]
        if weight_col:
            keep_columns.append(weight_col)
        df = copy.deepcopy(self.corpus[keep_columns])
        if sample_size:
            df = df.sample(n=sample_size).reset_index()
        if weight_col:
            df = df.dropna().reset_index()
        else:
            df = df.dropna(subset=[self.text_column, outcome_col]).reset_index()
        y = df[outcome_col]
        x = self.vectorizer.transform(df[self.text_column])
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
        )

    def kmeans_clusters(self, k=10):

        """
        A wrapper around :py:func:`pewanalytics.stats.clustering.compute_kmeans_clusters`. Will compute clusters of documents.
        The resulting cluster IDs for each document are saved in the TextDataFrame's ``corpus`` in a new column called
        "kmeans".

        :param k: The number of clusters to extract
        :type k: int

        Usage::

            >>> tdf.kmeans_clusters(5)
            KMeans: n_clusters 5, score is 0.019735248210503934
            KMeans clusters saved to self.corpus['kmeans']

            >>> df['kmeans'].value_counts()
            2    26
            3    15
            4    11
            0     5
            1     3
            Name: kmeans, dtype: int64
        """

        self.corpus["kmeans"] = compute_kmeans_clusters(
            self.tfidf, k=k, return_score=False
        )
        print("KMeans clusters saved to self.corpus['kmeans']")

    def hdbscan_clusters(self, min_cluster_size=100, min_samples=1):

        """
        A wrapper around :py:func:`pewanalytics.stats.clustering.compute_hdbscan_clusters`. Will compute clusters \
        of documents. The resulting cluster IDs for each document are saved in the TextDataFrame's ``corpus`` in a \
        new column called "hdbscan".

        :param min_cluster_size: The minimum number of documents that a cluster must contain.
        :type min_cluster_size: int
        :param min_samples: An HDBSCAN parameter; refer to the documentation for more information
        :type min_samples: int

        Usage::

            >>> tdf.hdbscan_clusters(min_cluster_size=10)
            HDBSCAN: n_clusters 2
            HDBSCAN clusters saved to self.corpus['hdbscan']
        """

        self.corpus["hdbscan"] = compute_hdbscan_clusters(
            self.tfidf, min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        print("HDBSCAN clusters saved to self.corpus['hdbscan']")

    def top_cluster_terms(self, cluster_col, min_size=50, top_n=10):

        """
        Extracts the top terms for each cluster, based on a column of cluster IDs saved to ``self.corpus``, using
        mutual information. Returns the ``top_n`` terms for each cluster.

        :param cluster_col: The name of the column that contains the document cluster IDs
        :type cluster_col: str
        :param min_size: Ignore clusters that have fewer than this number of documents
        :type min_size: int
        :param top_n: The number of top terms to identify for each cluster
        :type top_n: int
        :return: A dictionary; keys are the cluster IDs and values are the top terms for the cluster
        :rtype: dict

        Usage::

            >>> df_top_cluster = tdf.top_cluster_terms('kmeans', min_size=10)
            Cluster #2, 26 documents: ['constitution' 'union' 'states' 'friendly' 'liberal' 'revenue'
             'general government' 'confederacy' 'whilst' 'authorities']
            Cluster #4, 10 documents: ['shall strive' 'let sides' 'woe' 'offenses' 'breeze' 'war let'
             'nuclear weapons' 'learned live' 'mistakes' 'mr speaker']
            Cluster #0, 12 documents: ['activities' 'realization' 'interstate' 'wished' 'industrial' 'major'
             'counsel action' 'conditions' 'natural resources' 'eighteenth amendment']

        """

        dummies = pd.get_dummies(self.corpus[cluster_col], prefix=cluster_col)
        cluster_df = pd.concat([self.corpus, dummies], axis=1)

        terms = {}
        for cluster in cluster_df[cluster_col].unique():
            if (
                is_not_null(cluster)
                and len(cluster_df[cluster_df[cluster_col] == cluster]) >= min_size
            ):
                self.corpus["{}_{}".format(cluster_col, cluster)] = cluster_df[
                    "{}_{}".format(cluster_col, cluster)
                ]
                minfo = self.mutual_info("{}_{}".format(cluster_col, cluster))
                minfo = minfo.sort_values("MI1", ascending=False)[:top_n]
                del self.corpus["{}_{}".format(cluster_col, cluster)]
                minfo = minfo[minfo["MI1"] > 0].sort_values("MI1", ascending=False)[
                    :top_n
                ]
                terms[cluster] = minfo.index.values
                print(
                    "Cluster #{}, {} documents: {}".format(
                        cluster,
                        len(cluster_df[cluster_df[cluster_col] == cluster]),
                        minfo.index.values,
                    )
                )
        return terms

    def pca_components(self, k=20):

        """
        A wrapper around :py:func:`pewanalytics.stats.dimensionality_reduction.get_pca`.
        Saves the PCA components to self.corpus as new columns ('pca_1', 'pca_2', etc.),
        saves the top component for each document as self.corpus['pca'], and returns
        the features-component matrix.

        :param k: Number of dimensions to extract
        :type k: int
        :return: A :py:class:`pandas.DataFrame` of (features x components)

        Usage::

            >>> df_pca = tdf.pca_components(2)
            Decomposition explained variance ratio: 0.07488529151231405
            Component 0: ['america' 'today' 'americans' 'world' 'new' 'freedom' 'thank' 'nation'
             'god' 'journey']
            Component 1: ['america' 'make america' 'dreams' 'protected' 'obama' 'borders'
             'factories' 'american' 'transferring' 'stops']
            Top PCA dimensions saved as clusters to self.corpus['pca']

            >>> df.sample(5)
            	             speech	                                             text	year	21st_century	    pca_0      pca_1	  pca
            0	1789-Washington.txt	Fellow-Citizens of the Senate and of the House...	1789	0	            -0.129094	0.016984	pca_1
            21	1873-Grant.txt      Fellow-Citizens: Under Providence I have been ...	1873	0	            -0.097430	0.009559	pca_1
            49	1985-Reagan.txt  	Senator Mathias, Chief Justice Burger, Vice Pr...	1985	0	            0.163833	-0.020259	pca_0
            2	1797-Adams.txt    	When it was first perceived, in early times, t...	1797	0	            -0.140250	0.024844	pca_1
            20	1869-Grant.txt   	Citizens of the United States:    Your suffrag...	1869	0	            -0.114444	0.014419	pca_1
        """

        for col in self.corpus.columns:
            if col.startswith("pca_"):
                del self.corpus[col]
        components, documents = get_pca(
            self.tfidf, feature_names=self.vectorizer.get_feature_names(), k=k
        )
        for col in documents.columns:
            self.corpus[col] = documents[col]
        print("Top PCA dimensions saved as clusters to self.corpus['pca_'] columns")
        return components

    def lsa_components(self, k=20):

        """
        A wrapper around :py:func:`pewanalytics.stats.dimensionality_reduction.get_lsa`.
        Saves the LSA components to self.corpus as new columns ('lsa_1', 'lsa_2', etc.),
        saves the top component for each document as self.corpus['lsa'], and returns
        the features-component matrix

        :param k: Number of dimensions to extract
        :type k: int
        :return: A :py:class:`pandas.DataFrame` of (features x components)

        Usage::

            >>> df_lsa = tdf.lsa_components(2)
            Decomposition explained variance ratio: 0.04722850124656694
            Top features:
            Component 0: ['government' 'people' 'america' 'states' 'world' 'nation' 'shall'
             'country' 'great' 'peace']
            Component 1: ['america' 'today' 'americans' 'world' 'new' 'freedom' 'thank' 'nation'
             'god' 'journey']
            Top LSA dimensions saved as clusters to self.corpus['lsa_'] columns

            >>> df.sample(5)
            	            speech                                                 text    year	21st_century	lsa_0	   lsa_1	  lsa
            37	1937-Roosevelt.txt    When four years ago we met to inaugurate a Pre...    1937	           0 0.293068	0.040802	lsa_0
            8	1821-Monroe.txt       Fellow citizens, I shall not attempt to descri...    1821	           0 0.348465	-0.212382	lsa_0
            7	1817-Monroe.txt       I should be destitute of feeling if I was not ...    1817	           0 0.369249	-0.237231	lsa_0
            26	1893-Cleveland.txt    My Fellow citizens, in obedience of the mandat...    1893	           0 0.275778	-0.128497	lsa_0
            59	2017-Trump.txt        Chief Justice Roberts, President Carter, Presi...    2017	           1 0.342111	0.511687	lsa_1

        """

        for col in self.corpus.columns:
            if col.startswith("lsa_"):
                del self.corpus[col]
        components, documents = get_lsa(
            self.tfidf, feature_names=self.vectorizer.get_feature_names(), k=k
        )
        for col in documents.columns:
            self.corpus[col] = documents[col]
        print("Top LSA dimensions saved as clusters to self.corpus['lsa_'] columns")
        return components

    def get_top_documents(self, component_prefix="cluster", top_n=5):

        """
        Use after running :py:func:`pewanalytics.text.TextDataFrame.get_pca_components` or \
        :py:func:`pewanalytics.text.TextDataFrame.get_lsa_components`. Returns the ``top_n`` documents with \
        the highest scores for each components.

        :param component_prefix: 'lsa' or 'pca' (you must first run get_pca_components or get_lsa_components)
        :type component_prefix: str
        :param top_n: Number of documents to return for each component
        :type top_n: int
        :return: A dictionary where keys are the component, and values are the text values for the component's \
        ``top_n`` documents
        :rtype: dict

        Usage::

            >>> df_lsa_topdoc = tdf.get_top_documents("lsa")
            >>> {key: len(value) for key, value in lsa_topdoc.items()}
            {'lsa_0': 5, 'lsa_1': 4}

            >>> lsa_topdoc['lsa_1'][0]
            'Chief Justice Roberts, President Carter, President Clinton, President Bush, President Obama, fellow \
            Americans, and people of the world: Thank you.  We, the citizens of America...'

        """

        top_docs = {}
        for col in [
            c
            for c in self.corpus.columns
            if c.startswith("{}_".format(component_prefix))
        ]:
            docs = self.corpus[self.corpus[component_prefix] == col].sort_values(
                col, ascending=False
            )[:top_n]
            top_docs[col] = docs[self.text_column].values
        return top_docs

    def make_word_cooccurrence_matrix(
        self, normalize=False, min_frequency=10, max_frequency=0.5
    ):

        """
        Use to produce word co-occurrence matrices. Based on a helpful StackOverflow post:
        https://stackoverflow.com/questions/35562789/how-do-i-calculate-a-word-word-co-occurrence-matrix-with-sklearn

        :param normalize: If True, will be normalized
        :type normalize: bool
        :param min_frequency: The minimum document frequency required for a term to be included
        :type min_frequency: int
        :param max_frequency: The maximum proportion of documents containing a term allowed to include the term
        :type max_frequency: int
        :return: A matrix of (terms x terms) whose values indicate the number of documents in which two terms co-occurred

        Usage::

            >>> wcm = tdf.make_word_cooccurrence_matrix(min_frequency=25, normalize=True)
            # Find the top cooccurring pair of words
            >>> wcm.stack().index[np.argmax(wcm.values)]
            ('protection', 'policy')

        """

        text = self.corpus[self.text_column]
        cv = CountVectorizer(
            ngram_range=(1, 1),
            stop_words="english",
            min_df=min_frequency,
            max_df=max_frequency,
        )
        mat = cv.fit_transform(text)
        mat[
            mat > 0
        ] = (
            1
        )  # this makes sure that we're counting number of documents words have in common \
        # and not weighting by the frequency of one of the words in a single document, which can lead to spurious links
        names = cv.get_feature_names()
        mat = mat.T * mat  # compute the document-document matrix
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
        :type normalize: bool
        :return: A matrix of (documents x documents) whose values indicate the number of terms they had in common

        Usage::

            >>> dcm = tdf.make_document_cooccurrence_matrix(normalize=True)

            # Remove artifical duplicates and insert document names
            >>> dcm = dcm.iloc[:-2, :-2]
            >>> dcm.rename(columns=df['speech'][:-2],
                           index=df['speech'][:-2],
                           inplace=True)

            # Find documents with the highest coocurrence score
            >>> dcm.stack().index[np.argmax(dcm.values)]
            ('1793-Washington.txt', '1841-Harrison.txt')

        """

        text = self.corpus[self.text_column]
        cv = CountVectorizer(ngram_range=(1, 1), stop_words="english")
        mat = cv.fit_transform(text)
        mat[
            mat > 0
        ] = (
            1
        )  # this makes sure that we're counting number of words documents have in common \
        # and not weighting by the frequency of one of the words in a single document, which can lead to spurious links
        names = text.index
        mat = mat * mat.T  # compute the word-word matrix
        if normalize:
            diag = sp.diags(1.0 / mat.diagonal())
            mat = diag * mat
        mat.setdiag(0)
        matrix = pd.DataFrame(data=mat.todense(), columns=names, index=names)

        return matrix


def is_probable_stopword(word):
    """
    Determine if a word is likely to be a stop word (like a name of a person or location) by the following rules:

    1. Number of synset (words with similar meaning) is less than 3
    2. The min_depth (number of edges between a word and the top of the hierarchy) is > 5
    3. The number of lemma (similar to term definition in dictionary) is less than 2

    If more than one of these conditions is true, then this function will return False, because the word likely has
    one or more meanings in English and is likely to be more than just a proper name.

    This function was developed through trial and error, and your mileage may vary. It's intended to
    help you identify potential stopwords when extracting features from a database. For example, on one
    of our projects we wanted to remove names from our text data, and pulled a list of names from our
    database of politicians. However, some politicians have last names that are also common English words,
    like "White" and "Black" - and in those cases, we didn't want to add those to our list of stopwords.
    This function was useful in scanning through our list of names to identify names that we wanted to
    "whitelist".

    :param word: A word, usually a name of a person or location or something that you might want to add as a stopword
    :type word: string
    :return: Whether or not the word is (probably) a stopword aka a proper noun with no common English meaning
    :rtype: bool

    Usage::

        >>> is_probable_stopword("Chicago")
        True

        >>> is_probable_stopword("Chicago")
        False

        >>> is_probable_stopword("Orange")
        False

        >>> is_probable_stopword("Johnny")
        True

    """

    word = word.lower()
    synsets = wordnet.synsets(word)
    if not synsets or len(synsets) <= 1:
        return True
    else:
        total_synsets = len(synsets)
        min_depth = min([syn.min_depth() for syn in synsets])
        max_lemma_count = max(
            [sum([lemma.count() for lemma in syn.lemmas()]) for syn in synsets]
        )

        score = 0
        if total_synsets < 3:
            score += 1
        if min_depth >= 5:
            score += 1
        if max_lemma_count <= 2:
            score += 1

        if score > 1:
            return True

        else:
            return False
