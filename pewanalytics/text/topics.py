from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import gensim
import pandas as pd

from corextopic import corextopic
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import defaultdict

from pewtils import is_not_null, is_null, decode_text


class TopicModel(object):
    def __init__(
        self,
        df,
        text_col,
        num_topics=10,
        max_ngram_size=2,
        holdout_pct=0.25,
        use_tfidf=False,
        **vec_kwargs
    ):
        """
        Abstract class to provide a standardized way to train models from different libraries (sklearn and gensim).
        All implementations of this class will initialize a vectorizer and split the data into a train and test set if
        `holdout_pct` is provided.

        :param df: Pandas DataFrame
        :param text_col: Name of the column containing text
        :param num_topics: The number of topics to extract
        :param max_ngram_size: Maximum ngram size (2=bigrams, 3=trigrams, etc)
        :param holdout_pct: Proportion of the documents to hold out for goodness-of-fit scoring
        :param use_tfidf: Whether to use binary counts or a TF-IDF representation
        :param vec_kwargs: All remaining arguments get passed to TfidfVectorizer or CountVectorizer
        """

        self.df = df
        self.text_col = text_col
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
        self.train_features = self.get_features(self.train_df)
        self.test_features = self.get_features(self.test_df)
        self.model = None

        print("Initialized topic model with {} features".format(len(self.ngrams)))
        print(
            "{} training documents, {} testing documents".format(
                len(self.train_features), len(self.test_features)
            )
        )

    def get_features(self, df, keep_sparse=False):

        """
        Uses the trained vectorizer to process a dataframe and return a feature matrix.

        :param df: The DataFrame to vectorize (must have `self.text_col` in it)
        :param keep_sparse: Whether or not to keep the feature matrix in sparse format (default=False)
        :return: A Pandas DataFrame of features or a sparse matrix, depending on the value of `keep_sparse`
        """

        subset_df = df.dropna(subset=[self.text_col])
        features = self.vectorizer.transform(subset_df[self.text_col])
        if keep_sparse:
            return features
        else:
            return pd.DataFrame(
                features.todense(), columns=self.ngrams, index=subset_df.index
            )

    def fit(self, **kwargs):

        """
        Must be a function that fits a model and saves it to `self.model`
        """

        raise NotImplementedError()

    def get_score(self):

        """
        Must be a function that returns a dictionary with numeric goodness-of-fit scores
        """

        raise NotImplementedError()

    def get_document_topics(self, df):

        """
        Must be a function that takes a DataFrame and returns a document-topic DataFrame

        (rows=documents, columns=topics)
        :param df: The DataFrame to process (must have `self.text_col` in it)
        :return: A document-topic matrix
        """

        raise NotImplementedError()

    def get_topics(self, include_weights=False, top_n=10, **kwargs):

        """
        Must return a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

        raise NotImplementedError()

    def print_topics(self, include_weights=False, top_n=10):

        """
        Prints the top words for each topic from a trained model.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        """

        for i, topic in self.get_topics(
            include_weights=include_weights, top_n=top_n
        ).items():
            print("{}: {}".format(i, topic))


class ScikitLDATopicModel(TopicModel):

    """
    An implementation of the abstract `TopicModel` class that uses the scikit-learn LDA algorithm.
    """

    def fit(
        self,
        df=None,
        alpha=1.0,
        beta=1.0,
        learning_decay=0.7,
        learning_offset=50,
        learning_method="online",
        max_iter=500,
        batch_size=128,
        verbose=False,
    ):

        """
        Fits an LDA model using the sklearn implementation. For more information about available parameters, please
        refer to the official documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html

        :param df: The dataframe to train the model on (must contain `self.text_col`)
        :param alpha: Represents document-topic density. When values are higher, documents will be comprised of more
        topics; when values are lower, documents will be primarily comprised of only a few topics.
        :param beta: Represents topic-word density. When values are higher, topics will be comprised of more words;
        when values are lower, only a few words will be loaded onto each topic.
        :param learning_decay: See sklearn documentation.
        :param learning_offset: See sklearn documentation.
        :param learning_method: See sklearn documentation.
        :param max_iter: See sklearn documentation.
        :param batch_size: See sklearn documentation.
        :param verbose: See sklearn documentation.
        :return: None
        """

        if not self.model:
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                doc_topic_prior=alpha / float(self.num_topics),
                topic_word_prior=beta / float(self.num_topics),
                learning_decay=learning_decay,
                learning_method=learning_method,
                learning_offset=learning_offset,
                max_iter=max_iter,
                batch_size=batch_size,
                verbose=int(verbose),
            )
        if is_not_null(df):
            features = self.get_features(df)
        else:
            features = self.train_features
        self.model.fit(features)

    def get_score(self):

        """
        Returns the model's perplexity and log-likelihood scores, based on the holdout documents.

        :return: A dictionary with goodness-of-fit scores
        """

        if self.model:
            return {
                "perplexity": self.model.perplexity(self.test_features),
                "score": self.model.score(self.test_features),
            }

    def get_document_topics(self, df):

        """
        Takes a DataFrame and returns a document-topic DataFrame (rows=documents, columns=topics)

        :param df: The DataFrame to process (must have `self.text_col` in it)
        :return: A document-topic matrix
        """

        features = self.get_features(df)
        doc_topics = self.model.transform(features)
        topic_matrix = pd.DataFrame(
            doc_topics,
            columns=["topic_{}".format(i) for i in range(0, self.num_topics)],
            index=features.index,
        )
        return topic_matrix

    def get_topics(self, include_weights=False, top_n=10):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

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


class ScikitNMFTopicModel(TopicModel):

    """
    An implementation of the abstract `TopicModel` class that uses the scikit-learn non-negative matrix
    factorization (NMF) algorithm.
    """

    def fit(
        self, df=None, alpha=0.0, l1_ratio=0.5, tol=0.00001, max_iter=500, shuffle=True
    ):

        """
        Fits a NMF model using the sklearn implementation. For more information about available parameters, please
        refer to the official documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

        :param df: The dataframe to train the model on (must contain `self.text_col`)
        :param alpha: See sklearn documentation.
        :param l1_ratio: See sklearn documentation.
        :param tol: See sklearn documentation.
        :param max_iter: See sklearn documentation.
        :param shuffle: See sklearn documentation.
        :return: None
        """

        if not self.model:
            self.model = NMF(
                n_components=self.num_topics,
                alpha=alpha,
                l1_ratio=l1_ratio,
                tol=tol,
                max_iter=max_iter,
                shuffle=shuffle,
            )
        if is_not_null(df):
            features = self.get_features(df)
        else:
            features = self.train_features
        self.model.fit(features)

    def get_score(self):

        """
        Not currently implemented for NMF

        :return: Empty dictionary
        """

        if self.model:
            return {}

    def get_document_topics(self, df):

        """
        Takes a DataFrame and returns a document-topic DataFrame (rows=documents, columns=topics)

        :param df: The DataFrame to process (must have `self.text_col` in it)
        :return: A document-topic matrix
        """

        features = self.get_features(df)
        doc_topics = self.model.transform(features)
        topic_matrix = pd.DataFrame(
            doc_topics,
            columns=["topic_{}".format(i) for i in range(0, self.num_topics)],
            index=features.index,
        )

        topic_matrix = (
            topic_matrix.transpose() / topic_matrix.transpose().sum()
        ).transpose()  # to make each document sum to 1

        return topic_matrix

    def get_topics(self, include_weights=False, top_n=10):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

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


class GensimLDATopicModel(TopicModel):
    def __init__(self, *args, **kwargs):

        """
        An implementation of the abstract `TopicModel` class that uses the gensim LDA algorithm.
        """

        super(GensimLDATopicModel, self).__init__(*args, **kwargs)

        self.train_features = self.get_features(self.train_df, keep_sparse=True)
        self.test_features = self.get_features(self.test_df, keep_sparse=True)

    def fit(
        self,
        df=None,
        chunk_size=1000,
        passes=10,
        decay=0.8,
        offset=1,
        workers=2,
        alpha=None,
        beta="auto",
        use_multicore=False,
    ):

        """
        Fits an LDA model using gensim implementations. When `use_multicore` is set to `True`, the gensim multicore
        implementation will be used (https://radimrehurek.com/gensim/models/ldamulticore.html), otherwise the standard
        LDA implementation will be used (https://radimrehurek.com/gensim/models/ldamodel.html).
        For more information about available parameters, please refer to the official documentation.

        :param df: The dataframe to train the model on (must contain `self.text_col`)
        :param alpha: Represents document-topic density. When values are higher, documents will be comprised of more
        topics; when values are lower, documents will be primarily comprised of only a few topics. Gensim options are a
        bit different than sklearn though; refer to the documentation for the accepted values here.
        :param beta: Represents topic-word density. When values are higher, topics will be comprised of more words;
        when values are lower, only a few words will be loaded onto each topic. Gensim options are a bit different than
        sklearn though; refer to the documentation for the accepted values here. Gensim calls this parameter `eta`.
        :param chunk_size: See gensim documentation.
        :param passes: See gensim documentation.
        :param decay: See gensim documentation.
        :param offset: See gensim documentation.
        :param workers: Number of cores to use (if using multicore)
        :param use_multicore: Whether or not to use multicore
        :return: None
        """

        if not alpha:
            if use_multicore:
                alpha = "symmetric"
            else:
                alpha = "auto"
        vocab_dict = dict([(i, s) for i, s in enumerate(self.ngrams)])

        if not self.model:
            params = {
                "chunksize": chunk_size,
                "passes": passes,
                "decay": decay,
                "offset": offset,
                "num_topics": self.num_topics,
                "id2word": vocab_dict,
                "alpha": alpha,
                "eta": beta,
            }
            if use_multicore:
                model_class = gensim.models.ldamulticore.LdaMulticore
                params["workers"] = workers
            else:
                model_class = gensim.models.LdaModel
            self.model = model_class(**params)

        if is_not_null(df):
            features = self.get_features(df, keep_sparse=True)
        else:
            features = self.train_features
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
        self.model.update(matrix)

    def get_score(self):

        """
        Not currently implemented for LDA

        :return: Empty dictionary
        """

        return {}

    def get_document_topics(self, df, min_probability=0.0):

        """
        Takes a DataFrame and returns a document-topic DataFrame (rows=documents, columns=topics)

        :param df: The DataFrame to process (must have `self.text_col` in it)
        :param min_probability: Topics with a probability lower than this threshold will be filtered out
        :return: A document-topic matrix
        """

        features = self.get_features(df, keep_sparse=True)
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
        rows = []
        for index, bow in zip(df.dropna(subset=[self.text_col]).index, matrix):
            doc_topics = self.model.get_document_topics(
                bow, minimum_probability=min_probability
            )
            row = {"index": index}
            for topic, weight in doc_topics:
                row["topic_{}".format(topic)] = weight
            rows.append(row)
        return pd.DataFrame(rows)

    def get_topics(self, include_weights=False, top_n=10):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

        topics = defaultdict(list)
        for i in range(self.num_topics):
            for ngram, weight in self.model.show_topic(i, topn=top_n):
                if include_weights:
                    topics[i].append((ngram, weight))
                else:
                    topics[i].append(ngram)
        return topics


class GensimHDPTopicModel(TopicModel):
    def __init__(self, *args, **kwargs):

        """
        An implementation of the abstract `TopicModel` class that uses the gensim LDA algorithm.
        """

        super(GensimHDPTopicModel, self).__init__(*args, **kwargs)

        self.train_features = self.get_features(self.train_df, keep_sparse=True)
        self.test_features = self.get_features(self.test_df, keep_sparse=True)
        self.topic_ids = None
        self.num_topics = None
        if "num_topics" in kwargs.keys():
            raise Exception("You cannot specify the number of topics for an HDP model")

    def fit(
        self,
        df=None,
        max_chunks=None,
        max_time=None,
        chunksize=256,
        kappa=1.0,
        tau=64.0,
        T=150,
        K=15,
        alpha=1,
        beta=0.01,
        gamma=1,
        scale=1.0,
        var_converge=0.0001,
    ):

        """
        Fits an HDP model using the gensim implementation. Contrary to LDA and NMF, HDP attempts to auto-detect the
        correct number of topics. In practice, it actually fits `T` topics (default is 150) but many are extremely rare
        or occur only in a very few number of documents. To identify the topics that are actually useful, this function
        passes the original DataFrame through the trained model after fitting, and identifies topics that compose at
        least 1% of a document in at least 1% of all documents in the corpus. In other words, topics are thrown out if
        the number of documents they appear in at a rate of at least 1% are fewer than 1% of the total number of
        documents. Subsequent use of the model will only make use of topics that meet this threshold.

        For more information about available parameters, please refer to the official documentation:
        https://radimrehurek.com/gensim/models/hdpmodel.html

        :param df: The dataframe to train the model on (must contain `self.text_col`)
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
        :return:
        """

        vocab_dict = dict([(i, s) for i, s in enumerate(self.ngrams)])

        params = {
            "max_chunks": max_chunks,
            "max_time": max_time,
            "chunksize": chunksize,
            "kappa": kappa,
            "tau": tau,
            "T": T,
            "alpha": alpha,
            "gamma": gamma,
            "eta": beta,
            "scale": scale,
            "var_converge": var_converge,
            "K": K,
        }

        if is_not_null(df):
            features = self.get_features(df, keep_sparse=True)
        else:
            features = self.train_features
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)

        model_class = gensim.models.hdpmodel.HdpModel
        self.model = model_class(matrix, vocab_dict, **params)
        doc_topics = self.get_document_topics(self.df)

        topics = ((doc_topics >= 0.01).astype(int).mean() >= 0.01).astype(int)
        self.topic_ids = [
            int(col.split("_")[-1])
            for col in topics[topics == 1].index
            if col.startswith("topic_")
        ]
        self.num_topics = len(self.topic_ids)

    def get_score(self):

        """
        Not currently implemented for HDP

        :return: Empty dictionary
        """

        return {
            # "total_likelihood": self.model.evaluate_test_corpus(self.test_features)
        }

    def get_document_topics(self, df):

        """
        Takes a DataFrame and returns a document-topic DataFrame (rows=documents, columns=topics)

        :param df: The DataFrame to process (must have `self.text_col` in it)
        :return: A document-topic matrix
        """

        features = self.get_features(df, keep_sparse=True)
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
        rows = []
        for index, bow in zip(df.dropna(subset=[self.text_col]).index, matrix):
            doc_topics = self.model[bow]
            row = {"index": index}
            for topic, weight in doc_topics:
                if not self.topic_ids or topic in self.topic_ids:
                    row["topic_{}".format(topic)] = weight
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)

    def get_topics(self, include_weights=False, top_n=10):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

        topics = defaultdict(list)
        for i in self.topic_ids:
            for ngram, weight in self.model.show_topic(i, topn=top_n):
                if include_weights:
                    topics[i].append((ngram, weight))
                else:
                    topics[i].append(ngram)
        return topics


class CorExTopicModel(TopicModel):
    def __init__(self, *args, **kwargs):

        """
        An implementation of the CorEx topic modeling algorithm.
        """

        kwargs["use_tfidf"] = False
        super(CorExTopicModel, self).__init__(*args, **kwargs)

    def fit(self, df=None, anchors=None, anchor_strength=3):

        """
        Fits a CorEx topic model. Anchors can be provided in the form of a list of lists, with each item
        corresponding to a set of words to be used to seed a topic. For example:

        ```
        anchors=[
            ['cat', 'kitten'],
            ['dog', 'puppy']
        ]
        ```

        The list of anchors cannot be longer than the specified number of topics, and all of the words must
        exist in the vocabulary. The `anchor_strength` parameter determines the degree to which the model is able to
        override the suggested words based on the data; providing higher values are a way of "insisting" more strongly
        that the model keep the provided words together in a single topic.

        For more information about CorEx, refer to the official documentation:
        https://github.com/gregversteeg/corex_topic

        :param df: The dataframe to train the model on (must contain `self.text_col`)
        :param anchors: A list of lists that contain words that the model should try to group together into topics
        :param anchor_strength: The degree to which the provided anchors should be preserved regardless of the data
        :return:
        """

        if is_not_null(df):
            features = self.get_features(df, keep_sparse=True)
        else:
            features = self.get_features(self.train_df, keep_sparse=True)
        self.model = corextopic.Corex(n_hidden=self.num_topics)
        self.model.fit(
            features,
            words=self.ngrams,
            anchors=anchors,
            anchor_strength=anchor_strength,
        )

    def get_score(self):

        """
        Not currently implemented for CorEx

        :return: Empty dictionary
        """

        return {}

    def get_document_topics(self, df):

        """
        Takes a DataFrame and returns a document-topic DataFrame (rows=documents, columns=topics)

        :param df: The DataFrame to process (must have `self.text_col` in it)
        :return: A document-topic matrix
        """

        features = self.get_features(df, keep_sparse=True)
        doc_topics = self.model.transform(features)
        topic_matrix = pd.DataFrame(
            doc_topics,
            columns=["topic_{}".format(i) for i in range(0, self.num_topics)],
            index=df.index,
        )

        return topic_matrix

    def get_topics(self, include_weights=False, top_n=10):

        """
        Returns a list, equal in length to the number of topics, where each item is a list of words or word-weight
        tuples.

        :param include_weights: Whether or not to include weights along with the ngrams
        :param top_n: The number of words to include for each topic
        :return: A list of lists, where each item is a list of ngrams or ngram-weight tuples
        """

        topics = defaultdict(list)
        for topic_id, topic_ngrams in enumerate(self.model.get_topics(n_words=top_n)):
            for ngram, weight in topic_ngrams:
                if include_weights:
                    topics[topic_id].append((ngram, weight))
                else:
                    topics[topic_id].append(ngram)
        return topics
