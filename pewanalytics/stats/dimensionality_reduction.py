from __future__ import print_function
from builtins import zip
from builtins import range
import pandas as pd

from mca import MCA
from sklearn.decomposition import PCA, TruncatedSVD


def _decompose(
    features, decompose_class, feature_names=None, k=20, component_prefix="component"
):
    """
    Internal function used to break apart a set of features using a scikit-learn decomposition class and return \
    the resulting matrices.

    :param features: A :py:class:`pandas.DataFrame` or sparse matrix with rows are documents and columns are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :type feature_names: list
    :param k: Number of dimensions to extract
    :type k: int
    :param component_prefix: A prefix for the column names
    :type component_prefix: str
    :return: A tuple of two :py:class:`pandas.DataFrame`s, (features x components, documents x components)
    :rtype: tuple
    """

    model = decompose_class(n_components=k)
    try:
        features = pd.DataFrame(features.todense())
    except:
        pass
    model.fit(features)
    score = sum(model.explained_variance_ratio_)
    print("Decomposition explained variance ratio: {}".format(score))
    if not feature_names:
        feature_names = features.columns
    components = pd.DataFrame(model.components_, columns=feature_names).transpose()
    print("Top features:")
    for col in components.columns:
        print(
            "Component {}: {}".format(
                col, components.sort_values(col, ascending=False)[:10].index.values
            )
        )
    components.columns = [
        "{}_{}".format(component_prefix, c) for c in components.columns
    ]
    results = pd.DataFrame(model.transform(features), index=features.index)
    results.columns = components.columns
    results[component_prefix] = results.idxmax(axis=1)
    return (components, results)


def get_pca(features, feature_names=None, k=20):

    """
    Performs PCA on a set of features. This function expects input data where the rows are units \
    and columns are features.

    For more information about how PCA is implemented, visit the \
    `Scikit-Learn Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`__.

    :param features: A :py:class:`pandas.DataFrame` or sparse matrix where rows are units/observations and columns \
    are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :type feature_names: list
    :param k: Number of dimensions to extract
    :type k: int
    :return: A tuple of two :py:class:`pandas.DataFrame` s, (features x components, units x components)
    :rtype: tuple

    Usage::

        from pewanalytics.stats.dimensionality_reduction import get_pca
        from sklearn import datasets
        import pandas as pd

        df = pd.DataFrame(datasets.load_iris().data)

        >>> feature_weights, df_reduced  = get_pca(df, k=2)
        Decomposition explained variance ratio: 0.977685206318795
        Top features:
        Component 0: [2 0 3 1]
        Component 1: [1 0 3 2]

        >>> feature_weights
              pca_0     pca_1
        0  0.361387  0.656589
        1 -0.084523  0.730161
        2  0.856671 -0.173373
        3  0.358289 -0.075481

        >>> df_reduced.head()
              pca_0     pca_1    pca
        0 -2.684126  0.319397  pca_1
        1 -2.714142 -0.177001  pca_1
        2 -2.888991 -0.144949  pca_1
        3 -2.745343 -0.318299  pca_1
        4 -2.728717  0.326755  pca_1

    """

    return _decompose(
        features, PCA, feature_names=feature_names, k=k, component_prefix="pca"
    )


def get_lsa(features, feature_names=None, k=20):

    """
    Performs LSA on a set of features. This function expects input data where the rows are units \
    and columns are features.

    For more information about how LSA is implemented, visit the \
    `Scikit-Learn Documentation \
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`__.

    :param features: A :py:class:`pandas.DataFrame` or sparse matrix with rows are units/observations and columns \
    are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :type feature_names: list
    :param k: Number of dimensions to extract
    :type k: int
    :return: A tuple of two :py:class:`pandas.DataFrame` s, (features x components, documents x components)
    :rtype: tuple

    Usage::

        from pewanalytics.stats.dimensionality_reduction import get_lsa
        from sklearn import datasets
        import pandas as pd

        df = pd.DataFrame(datasets.load_iris().data)

        >>> feature_weights, df_reduced  = get_lsa(df, k=2)
        Decomposition explained variance ratio: 0.9772093692426493
        Top features:
        Component 0: [0 2 1 3]
        Component 1: [1 0 3 2]

        >>> feature_weights
              lsa_0     lsa_1
        0  0.751108  0.284175
        1  0.380086  0.546745
        2  0.513009 -0.708665
        3  0.167908 -0.343671

        >>> df_reduced.head()
              lsa_0     lsa_1    lsa
        0  5.912747  2.302033  lsa_0
        1  5.572482  1.971826  lsa_0
        2  5.446977  2.095206  lsa_0
        3  5.436459  1.870382  lsa_0
        4  5.875645  2.328290  lsa_0

    """

    return _decompose(
        features, TruncatedSVD, feature_names=feature_names, k=k, component_prefix="lsa"
    )


def correspondence_analysis(edges, n=1):

    """
    Performs correspondence analysis on a set of features.

    Most useful in the context of network analysis, where you might wish to, for example, \
    identify the underlying dimension in a network of Twitter users by using a matrix representing whether \
    or not they follow one another (when news and political accounts are included, the \
    underlying dimension often appears to approximate the left-right political spectrum.)

    :param edges: A :py:class:`pandas.DataFrame` of NxN where both the rows and columns are "nodes" and the values \
    are some sort of closeness or similarity measure (like a cosine similarity matrix)
    :param n: The number of dimensions to extract
    :type n: int
    :return: A :py:class:`pandas.DataFrame` where rows are the units and the columns correspond to the extracted \
    dimensions.

    Usage::

        from pewanalytics.stats.dimensionality_reduction import correspondence_analysis
        import nltk
        import pandas as pd
        from sklearn.metrics.pairwise import linear_kernel
        from sklearn.feature_extraction.text import TfidfVectorizer

        nltk.download("inaugural")
        df = pd.DataFrame([
            {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
        ])

        vec = TfidfVectorizer(min_df=10, max_df=.9).fit(df['text'])
        tfidf = vec.transform(df['text'])

        cosine_similarities = linear_kernel(tfidf)
        matrix = pd.DataFrame(cosine_similarities, columns=df['speech'])

        # Looks like the main source of variation in the language of inaugural speeches is time!

        >>> mca = correspondence_analysis(matrix)

        >>> mca.sort_values("mca_1").head()
                        node     mca_1
        57  1993-Clinton.txt -0.075508
        56    2017-Trump.txt -0.068168
        55  1997-Clinton.txt -0.061567
        54    1973-Nixon.txt -0.060698
        53     1989-Bush.txt -0.056305

        >>> mca.sort_values("mca_1").tail()
                       node     mca_1
        4    1877-Hayes.txt  0.040037
        3   1817-Monroe.txt  0.040540
        2     1845-Polk.txt  0.042847
        1   1849-Taylor.txt  0.050937
        0  1829-Jackson.txt  0.056201


    """

    mca_counts = MCA(edges)
    rows = []
    for r in sorted(
        zip(edges.columns, [m for m in mca_counts.fs_r(N=n)]),
        key=lambda x: x[1][0],
        reverse=True,
    ):
        row = {"node": r[0]}
        for i in range(n):
            try:
                row["mca_{}".format(i + 1)] = r[1][i]
            except:
                pass
        rows.append(row)
    mca = pd.DataFrame(rows)

    return mca
