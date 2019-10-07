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
    Used to break apart a set of features using a scikit-learn decomposition class and return the resulting matrices.

    :param features: A dataframe or sparse matrix with rows are documents and columns are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :param k: Number of dimensions to extract
    :param component_prefix: A prefix for the column names
    :return: A tuple of two dataframes, (features x components, documents x components)
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
    Performs PCA on a set of features.

    :param features: A dataframe or sparse matrix with rows are documents and columns are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :param k: Number of dimensions to extract
    :return: A tuple of two dataframes, (features x components, documents x components)
    """

    return _decompose(
        features, PCA, feature_names=feature_names, k=k, component_prefix="pca"
    )


def get_lsa(features, feature_names=None, k=20):

    """
    Performs LSA on a set of features.

    :param features: A dataframe or sparse matrix with rows are documents and columns are features
    :param feature_names: An optional list of feature names (for sparse matrices)
    :param k: Number of dimensions to extract
    :return: A tuple of two dataframes, (features x components, documents x components)
    """

    return _decompose(
        features, TruncatedSVD, feature_names=feature_names, k=k, component_prefix="lsa"
    )


def correspondence_analysis(edges, n=1):

    """
    :param edges: edges is a dataframe of NxN where both the rows and columns are "nodes" and the values are some sort
    of closeness or similarity measure (like a cosine similarity matrix)
    :param n: Number of dimensions to extract
    :return: A dataframe of the N dimensions
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
