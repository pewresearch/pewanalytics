from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import gensim
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import defaultdict

from pewtils import is_not_null, is_null, decode_text


class TopicModel(object):
    def __init__(self,
        df,
        text_col,
        num_topics=10,
        max_ngram_size=2,
        max_df=.4,
        min_df=10,
        max_features=10000,
        stop_words="english",
        holdout_pct=.25,
        tokenizer=None,
        analyzer="word",
        token_pattern=r'(?u)\b\w\w+\b',
        use_tfidf=False,
        **vec_kwargs
    ):

        self.df = df
        self.text_col = text_col
        self.num_topics = num_topics
        self.train_df = df.sample(int(round(len(df)*(1.0-holdout_pct))))
        self.train_df = self.train_df.dropna(subset=[self.text_col])
        self.test_df = df[~df.index.isin(self.train_df.index)]
        self.test_df = self.test_df.dropna(subset=[self.text_col])

        if use_tfidf: vec = TfidfVectorizer
        else: vec = CountVectorizer
        self.vectorizer = vec(
            ngram_range=(1, max_ngram_size),
            stop_words=stop_words,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            tokenizer=tokenizer,
            analyzer=analyzer,
            token_pattern=token_pattern,
            decode_error='ignore',
            **vec_kwargs
        )

        self.vectorizer = self.vectorizer.fit(self.train_df[self.text_col])
        self.ngrams = self.vectorizer.get_feature_names()
        self.train_features = self.get_features(self.train_df)
        self.test_features = self.get_features(self.test_df)
        self.model = None

        print( "Initialized topic model with {} features".format(len(self.ngrams)))
        print( "{} training documents, {} testing documents".format(len(self.train_features), len(self.test_features)))

    def get_features(self, df, keep_sparse=False):

        subset_df = df.dropna(subset=[self.text_col])
        features = self.vectorizer.transform(subset_df[self.text_col])
        if keep_sparse: return features
        else: return pd.DataFrame(features.todense(), columns=self.ngrams, index=subset_df.index)

    def fit(self, **kwargs):
        #TODO
        raise NotImplementedError()

    def get_score(self):
        raise NotImplementedError()

    def get_document_topics(self, df):
        raise NotImplementedError()

    def get_topics(self, **kwargs):
        raise NotImplementedError()

    def print_topics(self, include_weights=False, top_n=10):

        for i, topic in self.get_topics(include_weights=include_weights, top_n=top_n).items():
            print("{}: {}".format(i, topic))


class ScikitLDATopicModel(TopicModel):

    def fit(self,
        df=None,
        alpha=1.0,
        beta=1.0,
        learning_decay=.7,
        learning_offset=50,
        learning_method="online",
        max_iter=500,
        batch_size=128,
        verbose=False
    ):

        if not self.model:
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                doc_topic_prior=alpha/float(self.num_topics),
                topic_word_prior=beta/float(self.num_topics),
                learning_decay=learning_decay,
                learning_method=learning_method,
                learning_offset=learning_offset,
                max_iter=max_iter,
                batch_size=batch_size,
                verbose=int(verbose)
            )
        if is_not_null(df):
            features = self.get_features(df)
        else:
            features = self.train_features
        self.model.fit(features)

    def get_score(self):

        if self.model:
            return {
                "perplexity": self.model.perplexity(self.test_features),
                "score": self.model.score(self.test_features)
            }

    def get_document_topics(self, df):

        features = self.get_features(df)
        doc_topics = self.model.transform(features)
        topic_matrix = pd.DataFrame(doc_topics, columns=["topic_{}".format(i) for i in range(0, self.num_topics)], index=features.index)
        return topic_matrix

    def get_topics(self, include_weights=False, top_n=10):

        topic_features = self.model.components_
        topics = defaultdict(list)
        for topic_id, topic in enumerate(topic_features):
            top_ngram_index = sorted([(ngram_id, float(ngram_value)) for ngram_id, ngram_value in enumerate(topic)], key=lambda x: x[1], reverse=True)
            topics[topic_id] = [self.ngrams[ngram_id] if not include_weights else (self.ngrams[ngram_id], ngram_value) for ngram_id, ngram_value in top_ngram_index[:top_n]]
        return topics


class ScikitNMFTopicModel(TopicModel):

    def fit(self,
        df=None,
        alpha=0.0,
        l1_ratio=.5,
        tol=.00001,
        max_iter=500,
        shuffle=True
    ):

        if not self.model:
            self.model = NMF(
                n_components=self.num_topics,
                alpha=alpha,
                l1_ratio=l1_ratio,
                tol=tol,
                max_iter=max_iter,
                shuffle=shuffle
            )
        if is_not_null(df):
            features = self.get_features(df)
        else:
            features = self.train_features
        self.model.fit(features)

    def get_score(self):

        if self.model:
            return {}

    def get_document_topics(self, df):

        features = self.get_features(df)
        doc_topics = self.model.transform(features)
        topic_matrix = pd.DataFrame(doc_topics, columns=["topic_{}".format(i) for i in range(0, self.num_topics)], index=features.index)

        topic_matrix = (topic_matrix.transpose() / topic_matrix.transpose().sum()).transpose() # to make each document sum to 1

        return topic_matrix

    def get_topics(self, include_weights=False, top_n=10):

        topic_features = self.model.components_
        topics = defaultdict(list)
        for topic_id, topic in enumerate(topic_features):
            top_ngram_index = sorted([(ngram_id, float(ngram_value)) for ngram_id, ngram_value in enumerate(topic)], key=lambda x: x[1], reverse=True)
            topics[topic_id] = [self.ngrams[ngram_id] if not include_weights else (self.ngrams[ngram_id], ngram_value) for ngram_id, ngram_value in top_ngram_index[:top_n]]
        return topics


class GensimLDATopicModel(TopicModel):

    def __init__(self, *args, **kwargs):

        super(GensimLDATopicModel, self).__init__(*args, **kwargs)

        self.train_features = self.get_features(self.train_df, keep_sparse=True)
        self.test_features = self.get_features(self.test_df, keep_sparse=True)

    def fit(self,
        df=None,
        chunk_size=1000,
        passes=10,
        decay=.8,
        offset=1,
        workers=2,
        alpha="auto",
        beta="auto",
        use_multicore=False
    ):

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
                "eta": beta
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

        return {}

    def get_document_topics(self, df, min_probability=0.0):

        features = self.get_features(df, keep_sparse=True)
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
        rows = []
        for index, bow in zip(df.dropna(subset=[self.text_col]).index, matrix):
            doc_topics = self.model.get_document_topics(bow, minimum_probability=min_probability)
            row = {"index": index}
            for topic, weight in doc_topics:
                row["topic_{}".format(topic)] = weight
            rows.append(row)
        return pd.DataFrame(rows)

    def get_topics(self, include_weights=False, top_n=10):

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

        super(GensimHDPTopicModel, self).__init__(*args, **kwargs)

        self.train_features = self.get_features(self.train_df, keep_sparse=True)
        self.test_features = self.get_features(self.test_df, keep_sparse=True)

    def fit(self,
        df=None,
        max_chunks=None,
        max_time=None,
        chunksize=256,
        kappa=1.0,
        tau=64.0,
        T=150,
        alpha=1,
        gamma=1,
        eta=.01,
        scale=1.0,
        var_converge=.0001
    ):

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
            "eta": eta,
            "scale": scale,
            "var_converge": var_converge,
            "K": self.num_topics
        }

        if is_not_null(df):
            features = self.get_features(df, keep_sparse=True)
        else:
            features = self.train_features
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)

        model_class = gensim.models.hdpmodel.HdpModel
        self.model = model_class(matrix, vocab_dict, **params)

        self.num_topics = len(self.model.hdp_to_lda()[0])

    def get_score(self):
        #TODO
        return {
            # "total_likelihood": self.model.evaluate_test_corpus(self.test_features)
        }

    def get_document_topics(self, df, min_probability=0.0):

        features = self.get_features(df, keep_sparse=True)
        matrix = gensim.matutils.Sparse2Corpus(features, documents_columns=False)
        rows = []
        for index, bow in zip(df.dropna(subset=[self.text_col]).index, matrix):
            doc_topics = self.model[bow]
            row = {"index": index}
            for topic, weight in doc_topics:
                row["topic_{}".format(topic)] = weight
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)

    def get_topics(self, include_weights=False, top_n=10):

        topics = defaultdict(list)
        for i in range(self.num_topics):
            for ngram, weight in self.model.show_topic(i, topn=top_n):
                if include_weights:
                    topics[i].append((ngram, weight))
                else:
                    topics[i].append(ngram)
        return topics
