from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import gensim
import copy
import pandas as pd

from corextopic import corextopic
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import defaultdict

from pewtils import is_not_null


class TopicModel(object):

    """
    A wrapper around various topic modeling algorithms and libraries, intended to provide a standardized way to train \
    and apply models. When you initialize a ``TopicModel``, it will fit a vectorizer, and split the data into a train \
    and test set if ``holdout_pct`` is provided. For more information about the available implementations, refer to the \
    documentation for the ``fit()`` method below.

    :param df: A :py:class:`pandas.DataFrame`
    :param text_col: Name of the column containing text
    :type text_col: str
    :param method: The topic model implementation to use. Options are: sklearn_lda, sklearn_nmf, gensim_lda, \
    gensim_hdp, corex
    :type method: str
    :param num_topics: The number of topics to extract. Required for every method except ``gensim_hdp``.
    :type num_topics: int
    :param max_ngram_size: Maximum ngram size (2=bigrams, 3=trigrams, etc)
    :type max_ngram_size: int
    :param holdout_pct: Proportion of the documents to hold out for goodness-of-fit scoring
    :type holdout_pct: float
    :param use_tfidf: Whether to use binary counts or a TF-IDF representation
    :type use_tfidf: bool
    :param vec_kwargs: All remaining arguments get passed to TfidfVectorizer or CountVectorizer

    Usage::

        from pewanalytics.text.topics import TopicModel

        import nltk
        import pandas as pd
        nltk.download("movie_reviews")
        reviews = [{"fileid": fileid, "text": nltk.corpus.movie_reviews.raw(fileid)} for fileid in nltk.corpus.movie_reviews.fileids()]
        df = pd.DataFrame(reviews)

        >>> model = TopicModel(df, "text", "sklearn_nmf", num_topics=5, min_df=25, max_df=.5, use_tfidf=False)
        Initialized sklearn_nmf topic model with 3285 features
        1600 training documents, 400 testing documents

        >>> model.fit()

        >>> model.print_topics()
        0: bad, really, know, don, plot, people, scene, movies, action, scenes
        1: star, trek, star trek, effects, wars, star wars, special, special effects, movies, series
        2: jackie, films, chan, jackie chan, hong, master, drunken, action, tarantino, brown
        3: life, man, best, characters, new, love, world, little, does, great
        4: alien, series, aliens, characters, films, television, files, quite, mars, action

        >>> doc_topics = model.get_document_topics(df)

        >>> doc_topics
               topic_0   topic_1   topic_2   topic_3   topic_4
        0     0.723439  0.000000  0.000000  0.000000  0.000000
        1     0.289801  0.050055  0.000000  0.000000  0.000000
        2     0.375149  0.000000  0.030691  0.059088  0.143679
        3     0.152961  0.010386  0.000000  0.121412  0.015865
        4     0.294005  0.100426  0.000000  0.137630  0.051241
        ...        ...       ...       ...       ...       ...
        1995  0.480983  0.070431  0.135178  0.256951  0.000000
        1996  0.139986  0.000000  0.000000  0.107430  0.000000
        1997  0.141545  0.005990  0.081986  0.387859  0.057025
        1998  0.029228  0.023342  0.043713  0.280877  0.107551
        1999  0.044863  0.000000  0.000000  0.718677  0.000000

    """

    def __init__(
        self,
        df,
        text_col,
        method,
        num_topics=None,
        max_ngram_size=2,
        holdout_pct=0.25,
        use_tfidf=False,
        **vec_kwargs
    ):

        self.df = df
        self.text_col = text_col
        self.method = method
        self.num_topics = num_topics
        self.train_df = df.sample(int(round(len(df) * (1.0 - holdout_pct))))
        self.train_df = self.train_df.dropna(subset=[self.text_col])
        self.test_df = df[~df.index.isin(self.train_df.index)]
        self.test_df = self.test_df.dropna(subset=[self.text_col])
        if "stop_words" not in vec_kwargs:
            vec_kwargs["stop_words"] = "english"

        if use_tfidf:
            vec = TfidfVectorizer
        else:
            vec = CountVectorizer
        self.vectorizer = vec(
            ngram_range=(1, max_ngram_size), decode_error="ignore", **vec_kwargs
        )

        self.vectorizer = self.vectorizer.fit(self.train_df[self.text_col])
        self.ngrams = self.vectorizer.get_feature_names()
        if self.method in ["gensim_lda", "gensim_hdp"]:
            self.train_features = self.get_features(self.train_df, keep_sparse=True)
            self.test_features = self.get_features(self.test_df, keep_sparse=True)
            if self.method == "gensim_hdp":
                self.topic_ids = None
                if num_topics:
                    raise Exception(
                        "You cannot specify the number of topics for an HDP model"
                    )
        else:
            self.train_features = self.get_features(self.train_df)
            self.test_features = self.get_features(self.test_df)

        self.model = None

        print(
            "Initialized {} topic model with {} features".format(
                self.method, len(self.ngrams)
            )
        )
        try:
            print(
                "{} training documents, {} testing documents".format(
                    len(self.train_features), len(self.test_features)
                )
            )
        except TypeError:
            print(
                "{} training documents, {} testing documents".format(
                    self.train_features.shape[0], self.test_features.shape[0]
                )
            )

    def get_features(self, df, keep_sparse=False):

        """
        Uses the trained vectorizer to process a :py:class:`pandas.DataFrame` and return a feature matrix.

        :param df: The :py:class:`pandas.DataFrame` to vectorize (must have ``self.text_col`` in it)
        :param keep_sparse: Whether or not to keep the feature matrix in sparse format (default=False)
        :type keep_sparse: bool
        :return: A :py:class:`pandas.DataFrame` of features or a sparse matrix, depending on the value of \
        ``keep_sparse``
        """

        subset_df = df.dropna(subset=[self.text_col])
        features = self.vectorizer.transform(subset_df[self.text_col])
        if keep_sparse:
            return features
        else:
            return pd.DataFrame(
                features.todense(), columns=self.ngrams, index=subset_df.index
            )

    def get_fit_params(self, **kwargs):

        """
        Internal helper function to set defaults depending on the specified model.

        :param kwargs: Arguments passed to ``self.fit()``
        :return: Arguments to pass to the model
        """

        defaults = {
            "sklearn_lda": {
                "alpha": 1.0,
                "beta": 1.0,
                "learning_decay": 0.7,
                "learning_offset": 50,
                "learning_method": "online",
                "max_iter": 500,
                "batch_size": 128,
                "verbose": False,
            },
            "sklearn_nmf": {
                "alpha": 0.0,
                "l1_ratio": 0.5,
                "tol": 0.00001,
                "max_iter": 500,
                "shuffle": True,
            },
            "gensim_lda": {
                "chunksize": 1000,
                "passes": 10,
                "decay": 0.8,
                "offset": 1,
                "workers": 2,
                "alpha": None,
                "beta": "auto",
                "use_multicore": False,
            },
            "gensim_hdp": {
                "max_chunks": None,
                "max_time": None,
                "chunksize": 256,
                "kappa": 1.0,
                "tau": 64.0,
                "T": 150,
                "K": 15,
                "alpha": 1,
                "beta": 0.01,
                "gamma": 1,
                "scale": 1.0,
                "var_converge": 0.0001,
            },
            "corex": {"anchors": [], "anchor_strength": 3},
        }

        for k, v in kwargs.items():
            if k not in defaults[self.method].keys():
                raise Exception(
                    "Unknown keyword argument for method '{}': {}. Accepted parameters are: {}".format(
                        self.method, k, defaults[self.method].keys()
                    )
                )
        fit_params = copy.deepcopy(defaults[self.method])
        fit_params.update(kwargs)

        if self.method == "sklearn_lda":
            fit_params["verbose"] = int(fit_params["verbose"])
            if "alpha" in fit_params.keys():
                fit_params["doc_topic_prior"] = fit_params["alpha"] / float(
                    self.num_topics
                )
                del fit_params["alpha"]
            if "beta" in fit_params.keys():
                fit_params["topic_word_prior"] = fit_params["beta"] / float(
                    self.num_topics
                )
                del fit_params["beta"]

        if self.method == "gensim_lda":
            if not fit_params["alpha"]:
                if fit_params["use_multicore"]:
                    fit_params["alpha"] = "symmetric"
                else:
                    fit_params["alpha"] = "auto"

        if self.method in ["gensim_lda", "gensim_hdp"]:
            if "beta" in fit_params.keys():
                fit_params["eta"] = fit_params["beta"]
                del fit_params["beta"]

        return fit_params

    def fit(self, df=None, **kwargs):

        """
        Fits a model using the method specified when initializing the ``TopicModel``. Details on model-specific \
        parameters are below:

        **sklearn_lda**

        Fits a model using :py:class:`sklearn.decomposition.LatentDirichletAllocation`. For more information on \
        available parameters, please refer to the official documentation: \
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html

        :param df: The :py:class:`pandas.DataFrame` to train the model on (must contain ``self.text_col``)
        :param alpha: Represents document-topic density. When values are higher, documents will be comprised of more \
        topics; when values are lower, documents will be primarily comprised of only a few topics. This parameter is \
        used instead of the doc_topic_prior sklearn parameter, and will be passed along to sklearn using the formula: \
        ``doc_topic_prior = alpha / num_topics``
        :param beta: Represents topic-word density. When values are higher, topics will be comprised of more words; \
        when values are lower, only a few words will be loaded onto each topic. This parameter is used instead of the \
        topic_word_prior sklearn parameter, and will be passed along to sklearn using the formula: \
        ``topic_word_prior = beta / num_topics``.
        :param learning_decay: See sklearn documentation.
        :param learning_offset: See sklearn documentation.
        :param learning_method: See sklearn documentation.
        :param max_iter: See sklearn documentation.
        :param batch_size: See sklearn documentation.
        :param verbose: See sklearn documentation.

        **sklearn_nmf**

        Fits a model using :py:class:`sklearn.decomposition.NMF`. For more information on available parameters, \
        please refer to the official documentation: \
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

        :param df: The :py:class:`pandas.DataFrame` to train the model on (must contain ``self.text_col``)
        :param alpha: See sklearn documentation.
        :param l1_ratio: See sklearn documentation.
        :param tol: See sklearn documentation.
        :param max_iter: See sklearn documentation.
        :param shuffle: See sklearn documentation.

        **gensim_lda**

        Fits an LDA model using :py:class:`gensim.models.LdaModel` or \
        :py:class:`gensim.models.ldamulticore.LdaMulticore`. \
        When ``use_multicore`` is set to True, the multicore implementation will be used, otherwise the standard \
        LDA implementation will be used. \
        For more information on available parameters, please refer to the official documentation below:

            - use_multicore=True: https://radimrehurek.com/gensim/models/ldamulticore.html
            - use_multicore=False: https://radimrehurek.com/gensim/models/ldamodel.html

        :param df: The :py:class:`pandas.DataFrame` to train the model on (must contain ``self.text_col``)
        :param alpha: Represents document-topic density. When values are higher, documents will be comprised of \
        more topics; when values are lower, documents will be primarily comprised of only a few topics. Gensim \
        options are a bit different than sklearn though; refer to the documentation for the accepted values here.
        :param beta: Represents topic-word density. When values are higher, topics will be comprised of more words; \
        when values are lower, only a few words will be loaded onto each topic. Gensim options are a bit different \
        than sklearn though; refer to the documentation for the accepted values here. Gensim calls this parameter \
        ``eta``. We renamed it to be consistent with the sklearn implementations.
        :param chunksize: See gensim documentation.
        :param passes: See gensim documentation.
        :param decay: See gensim documentation.
        :param offset: See gensim documentation.
        :param workers: Number of cores to use (if using multicore)
        :param use_multicore: Whether or not to use multicore

        **gensim_hdp**

        Fits an HDP model using the gensim implementation. Contrary to LDA and NMF, HDP attempts to auto-detect the
        correct number of topics. In practice, it actually fits ``T`` topics (default is 150) but many are extremely rare
        or occur only in a very few number of documents. To identify the topics that are actually useful, this function
        passes the original :py:class:`pandas.DataFrame` through the trained model after fitting, and identifies \
        topics that compose at least 1% of a document in at least 1% of all documents in the corpus. In other words, \
        topics are thrown out if the number of documents they appear in at a rate of at least 1% are fewer than 1% of \
        the total number of documents. Subsequent use of the model will only make use of topics that meet this \
        threshold. For more information on available parameters, please refer to the official documentation: \
        https://radimrehurek.com/gensim/models/hdpmodel.html

        :param df: The :py:class:`pandas.DataFrame` to train the model on (must contain ``self.text_col``)
        :param max_chunks: See gensim documentation.
        :param max_time: See gensim documentation.
        :param chunksize: See gensim documentation.
        :param kappa: See gensim documentation.
        :param tau: See gensim documentation.
        :param T: See gensim documentation.
        :param K: See gensim documentation.
        :param alpha: See gensim documentation.
        :param beta: See gensim documentation.
        :param gamma: See gensim documentation.
        :param scale: See gensim documentation.
        :param var_converge: See gensim documentation.

        **corex**

        Fits a CorEx topic model. Anchors can be provided in the form of a list of lists, with each item
        corresponding to a set of words to be used to seed a topic. For example:

        .. code-block:: python

            anchors=[
                ['cat', 'kitten'],
                ['dog', 'puppy']
            ]

        The list of anchors cannot be longer than the specified number of topics, and all of the words must
        exist in the vocabulary. The ``anchor_strength`` parameter determines the degree to which the model is able to
        override the suggested words based on the data; providing higher values are a way of "insisting" more strongly
        that the model keep the provided words together in a single topic. For more information on available \
        parameters, please refer to the official documentation: https://github.com/gregversteeg/corex_topic

        :param df: The :py:class:`pandas.DataFrame` to train the model on (must contain ``self.text_col``)
        :param anchors: A list of lists that contain words that the model should try to group together into topics
        :param anchor_strength: The degree to which the provided anchors should be preserved regardless of the data

        """

        fit_params = self.get_fit_params(**kwargs)

        if self.method in ["sklearn_lda", "sklearn_nmf"]:

            if self.method == "sklearn_lda":
                self.model = LatentDirichletAllocation(
                    n_components=self.num_topics, **fit_params
                )
            if self.method == "sklearn_nmf":
                self.model = NMF(n_components=self.num_topics, **fit_params)

            if is_not_null(df):
                features = self.get_features(df)
            else:
                features = self.train_features
            self.model.fit(features)

        elif self.method in ["gensim_lda", "gensim_hdp"]:

            vocab_dict = dict([(i, s) for i, s in enumerate(self.ngrams)])
            if is_not_null(df):
                features = self.get_features(df, keep_sparse=True)
            else:
                features = self.train_features
            matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)

            if self.method == "gensim_lda":
                fit_params["num_topics"] = self.num_topics
                fit_params["id2word"] = vocab_dict
                if fit_params["use_multicore"]:
                    model_class = gensim.models.ldamulticore.LdaMulticore
                else:
                    model_class = gensim.models.LdaModel
                    del fit_params["workers"]
                del fit_params["use_multicore"]
                self.model = model_class(**fit_params)
                self.model.update(matrix)
            elif self.method == "gensim_hdp":
                model_class = gensim.models.hdpmodel.HdpModel
                self.model = model_class(matrix, vocab_dict, **fit_params)
                doc_topics = self.get_document_topics(self.df)
                topics = ((doc_topics >= 0.01).astype(int).mean() >= 0.01).astype(int)
                self.topic_ids = [
                    int(col.split("_")[-1])
                    for col in topics[topics == 1].index
                    if col.startswith("topic_")
                ]
                self.num_topics = len(self.topic_ids)

        elif self.method == "corex":

            if is_not_null(df):
                features = self.get_features(df, keep_sparse=True)
            else:
                features = self.get_features(self.train_df, keep_sparse=True)
            self.model = corextopic.Corex(n_hidden=self.num_topics)
            self.model.fit(features, words=self.ngrams, **fit_params)

    def get_score(self):

        """
        Returns goodness-of-fit scores for certain models, based on the holdout documents.

        .. note:: The following scores are available for the following methods:

                - perplexity: (sklearn_lda only) The model's perplexity
                - score: (sklearn_lda only) The model's log-likelihood score
                - total_correlation: (corex only) The model's total correlation score

        :return: A dictionary with goodness-of-fit scores
        :rtype: dict

        """

        if self.model:
            if self.method == "sklearn_lda":
                return {
                    "perplexity": self.model.perplexity(self.test_features),
                    "score": self.model.score(self.test_features),
                }
            elif self.method == "corex":
                return {"total_correlation": self.model.tc}
            else:
                return {}

    def get_document_topics(self, df, **kwargs):

        """
        Takes a :py:class:`pandas.DataFrame` and returns a document-topic :py:class:`pandas.DataFrame` \
        (rows=documents, columns=topics)

        :param df: The :py:class:`pandas.DataFrame` to process (must have ``self.text_col`` in it)
        :param min_probability: (gensim_lda use_multicore=False only) Topics with a probability lower than this \
        threshold will be filtered out (Default=0.0)
        :type min_probability: float
        :return: A document-topic matrix
        """

        if self.method in ["sklearn_lda", "sklearn_nmf"]:

            features = self.get_features(df)
            doc_topics = self.model.transform(features)
            topic_matrix = pd.DataFrame(
                doc_topics,
                columns=["topic_{}".format(i) for i in range(0, self.num_topics)],
                index=features.index,
            )
            return topic_matrix

        elif self.method in ["gensim_lda", "gensim_hdp"]:

            features = self.get_features(df, keep_sparse=True)
            matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
            rows = []
            for index, bow in zip(df.dropna(subset=[self.text_col]).index, matrix):
                if self.method == "gensim_lda":
                    if "min_probability" not in kwargs:
                        kwargs["min_probability"] = 0.0
                    try:
                        doc_topics = self.model.get_document_topics(bow, **kwargs)
                    except TypeError:
                        del kwargs["min_probability"]
                        doc_topics = self.model.get_document_topics(bow, **kwargs)
                elif self.method == "gensim_hdp":
                    doc_topics = self.model[bow]
                row = {"index": index}
                for topic, weight in doc_topics:
                    if self.method == "gensim_lda" or (
                        not self.topic_ids or topic in self.topic_ids
                    ):
                        row["topic_{}".format(topic)] = weight
                rows.append(row)
            df = pd.DataFrame(rows).fillna(0)
            df = df.set_index(df["index"])
            del df["index"]
            return df

        elif self.method == "corex":

            features = self.get_features(df, keep_sparse=True)
            doc_topics = self.model.transform(features)
            topic_matrix = pd.DataFrame(
                doc_topics,
                columns=["topic_{}".format(i) for i in range(0, self.num_topics)],
                index=df.index,
            )
            return topic_matrix

    def get_topics(self, include_weights=False, top_n=10, **kwargs):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :type include_weights: bool
        :param top_n: The number of words to include for each topic
        :type top_n: init
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

        if self.method in ["sklearn_lda", "sklearn_nmf"]:

            topic_features = self.model.components_
            topics = defaultdict(list)
            for topic_id, topic in enumerate(topic_features):
                top_ngram_index = sorted(
                    [
                        (ngram_id, float(ngram_value))
                        for ngram_id, ngram_value in enumerate(topic)
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                topics[topic_id] = [
                    self.ngrams[ngram_id]
                    if not include_weights
                    else (self.ngrams[ngram_id], ngram_value)
                    for ngram_id, ngram_value in top_ngram_index[:top_n]
                ]
            return topics

        elif self.method in ["gensim_lda", "gensim_hdp"]:

            topics = defaultdict(list)
            if self.method == "gensim_hdp":
                topic_ids = self.topic_ids
            else:
                topic_ids = range(self.num_topics)
            for i in topic_ids:
                for ngram, weight in self.model.show_topic(i, topn=top_n):
                    if include_weights:
                        topics[i].append((ngram, weight))
                    else:
                        topics[i].append(ngram)
            return topics

        elif self.method == "corex":

            topics = defaultdict(list)
            for topic_id, topic_ngrams in enumerate(
                self.model.get_topics(n_words=top_n)
            ):
                for ngram, weight, _ in topic_ngrams:
                    if include_weights:
                        topics[topic_id].append((ngram, weight))
                    else:
                        topics[topic_id].append(ngram)
            return topics

    def print_topics(self, include_weights=False, top_n=10):

        """
        Prints the top words for each topic from a trained model.

        :param include_weights: Whether or not to include weights along with the ngrams
        :type include_weights: bool
        :param top_n: The number of words to include for each topic
        :type top_n: int
        """

        for i, topic in self.get_topics(
            include_weights=include_weights, top_n=top_n
        ).items():
            print("{}: {}".format(i, ", ".join(topic)))
