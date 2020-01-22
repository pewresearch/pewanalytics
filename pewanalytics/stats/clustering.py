from __future__ import print_function
import hdbscan
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel


def compute_kmeans_clusters(features, k=10, return_score=False):
    """
    Uses K-Means to cluster an arbitrary set of features.

    :param features: TF-IDF sparse matrix or pandas DataFrame
    :param k: int, number of clusters
    :param return_score: bool, default=False; if True, returns a tuple with the cluster \
    assignments and the silhouette score of the clustering
    :return: series corresponding to the cluster labels for each row
    """

    km = KMeans(n_clusters=k)
    labels = km.fit_predict(features)
    silhouette_avg = silhouette_score(features, labels)
    print("KMeans: n_clusters {}, score is {}".format(k, silhouette_avg))
    if not return_score:
        return km.labels_.tolist()
    else:
        return (km.labels_.tolist(), silhouette_avg)


def compute_hdbscan_clusters(features, min_cluster_size=100, min_samples=1):
    """
    Uses HDBSCAN to identify the best number of clusters and map each unit to one.

    | Rows are the units, and columns are features. Returns a series with the cluster labels \
    for each row.

    :param features: TF-IDF sparse matrix or a Pandas DataFrame
    :param min_cluster_size: int - minimum number of documents/units that can exist in a cluster.
    :param min_samples: Minimum number of samples to draw (see HDBSCAN documentation for more)
    :return: Series with the cluster ID that each row belongs to
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    clusterer.fit(features)
    print("HDBSCAN: n_clusters {}".format(clusterer.labels_.max() + 1))
    return clusterer.labels_
