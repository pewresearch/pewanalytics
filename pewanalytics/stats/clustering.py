from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_kmeans_clusters(features, k=10, return_score=False):
    """
    Uses K-Means to cluster an arbitrary set of features. This function expects input data where the rows are units \
    and columns are features.

    :param features: TF-IDF sparse matrix or :py:class:`pandas.DataFrame`
    :param k: The number of clusters to extract
    :type k: int
    :param return_score: If True, the function returns a tuple with the cluster \
    assignments and the silhouette score of the clustering; otherwise the function just returns a list of cluster \
    labels for each row. (Default=False)
    :type return_score: bool
    :return: A list with the cluster label for each row, or a tuple containing the \
    labels followed by the silhouette score of the K-Means model.
    :rtype: list

    Usage::

        from pewanalytics.stats.clustering import compute_kmeans_clusters
        from sklearn import datasets
        import pandas as pd

        # The iris dataset is a common example dataset included in scikit-learn with 3 main clusters
        # Let's see if we can find them
        df = pd.DataFrame(datasets.load_iris().data)

        >>> df['cluster'] = compute_kmeans_clusters(df, k=3)
        KMeans: n_clusters 3, score is 0.5576853964035263

        >>> df['cluster'].value_counts()
        1    62
        0    50
        2    38
        Name: cluster, dtype: int64

    """

    km = KMeans(n_clusters=k)
    labels = km.fit_predict(features)
    silhouette_avg = silhouette_score(features, labels)
    print("KMeans: n_clusters {}, score is {}".format(k, silhouette_avg))
    if not return_score:
        return km.labels_.tolist()
    else:
        return (km.labels_.tolist(), silhouette_avg)


def compute_hdbscan_clusters(features, min_cluster_size=100, min_samples=1, **kwargs):
    """
    Uses HDBSCAN* to identify the best number of clusters and map each unit to one. This function expects input data \
    where the rows are units and columns are features. Additional keyword arguments are passed to HDBSCAN. Check \
    out the official documentation for more: https://hdbscan.readthedocs.io/en/latest

    :param features: TF-IDF sparse matrix or :py:class:`pandas.DataFrame`
    :param min_cluster_size: int - minimum number of documents/units that can exist in a cluster.
    :type min_cluster_size: int
    :param min_samples: Minimum number of samples to draw (see HDBSCAN documentation for more)
    :type min_samples: int
    :param kwargs: Additional HDBSCAN parameters: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    :return: A list with the cluster label for each row

    Usage::

        from pewanalytics.stats.clustering import compute_hdbscan_clusters
        from sklearn import datasets
        import pandas as pd

        df = pd.DataFrame(datasets.load_iris().data)

        >>> df['cluster'] = compute_hdbscan_clusters(df, min_cluster_size=10)
        HDBSCAN: n_clusters 2

        >>> df['cluster'].value_counts()
        1    100
        0     50
        Name: cluster, dtype: int64

    """

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs
    )
    clusterer.fit(features)
    print("HDBSCAN: n_clusters {}".format(clusterer.labels_.max() + 1))
    return clusterer.labels_
