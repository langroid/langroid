import logging
from collections import Counter
from typing import Callable, List, Tuple

import faiss
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from llmagent.mytypes import Document

logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("faiss-cpu").setLevel(logging.ERROR)


def find_optimal_clusters(X, max_clusters, threshold=0.1):
    """
    Find the optimal number of clusters for FAISS K-means using the Elbow Method.

    Args:
        X (np.ndarray): A 2D NumPy array of data points.
        max_clusters (int): The maximum number of clusters to try.
        threshold (float): Threshold for the rate of change in inertia values.
        Defaults to 0.1.

    Returns:
        int: The optimal number of clusters.
    """
    inertias = []
    max_clusters = min(max_clusters, X.shape[0])
    cluster_range = range(1, max_clusters + 1)

    for nclusters in cluster_range:
        kmeans = faiss.Kmeans(X.shape[1], nclusters, niter=20, verbose=False)
        kmeans.train(X)
        centroids = kmeans.centroids
        distances = np.sum(np.square(X[:, None] - centroids), axis=-1)
        inertia = np.sum(np.min(distances, axis=-1))
        inertias.append(inertia)

    # Calculate the rate of change in inertia values
    rate_of_change = [
        abs((inertias[i + 1] - inertias[i]) / inertias[i])
        for i in range(len(inertias) - 1)
    ]

    # Find the optimal number of clusters based on the rate of change threshold
    optimal_clusters = 1
    for i, roc in enumerate(rate_of_change):
        if roc < threshold:
            optimal_clusters = i + 1
            break

    return optimal_clusters


def densest_clusters(
    embeddings: List[np.ndarray], k: int = 5
) -> List[Tuple[np.ndarray, int]]:
    """
    Find the top k densest clusters in the given list of embeddings using FAISS K-means.
    See here:
    'https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%
     2C-PCA%2C-quantization'

    Args:
        embeddings (List[np.ndarray]): A list of embedding vectors.
        k (int, optional): The number of densest clusters to find. Defaults to 5.

    Returns:
        List[Tuple[np.ndarray, int]]: A list of representative vectors and their indices
                                      from the k densest clusters.
    """
    # Convert the list of embeddings to a NumPy array
    X = np.vstack(embeddings)

    # FAISS K-means clustering
    ncentroids = find_optimal_clusters(X, max_clusters=2 * k, threshold=0.1)
    k = min(k, ncentroids)
    niter = 20
    verbose = True
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose)
    kmeans.train(X)

    # Get the cluster centroids
    centroids = kmeans.centroids

    # Find the nearest neighbors of the centroids in the original embeddings
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors(centroids)

    # Sort the centroids by their nearest neighbor distances
    sorted_centroids_indices = np.argsort(distances, axis=0).flatten()

    # Select the top k densest clusters
    densest_clusters_indices = sorted_centroids_indices[:k]

    # Get the representative vectors and their indices from the densest clusters
    representative_vectors = [
        (idx, embeddings[idx]) for idx in densest_clusters_indices
    ]

    return representative_vectors


def densest_clusters_DBSCAN(
    embeddings: np.ndarray, k: int = 10
) -> List[Tuple[int, np.ndarray]]:
    """
    Find the representative vector and corresponding index from each of the k densest
    clusters in the given embeddings.

    Args:
        embeddings (np.ndarray): A NumPy array of shape (n, d), where n is the number
                                 of embedding vectors and d is their dimensionality.
        k (int): Number of densest clusters to find.

    Returns:
        List[Tuple[int, np.ndarray]]: A list of tuples containing the index and
                                      representative vector for each of the k densest
                                      clusters.
    """

    # Normalize the embeddings if necessary
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Choose a clustering algorithm (DBSCAN in this case)
    # Tune eps and min_samples for your use case
    dbscan = DBSCAN(eps=4, min_samples=5)

    # Apply the clustering algorithm
    cluster_labels = dbscan.fit_predict(embeddings_normalized)

    # Compute the densities of the clusters
    cluster_density = Counter(cluster_labels)

    # Sort clusters by their density
    sorted_clusters = sorted(cluster_density.items(), key=lambda x: x[1], reverse=True)

    # Select top-k densest clusters
    top_k_clusters = sorted_clusters[:k]

    # Find a representative vector for each cluster
    representatives = []
    for cluster_id, _ in top_k_clusters:
        if cluster_id == -1:
            continue  # Skip the noise cluster (label -1)
        indices = np.where(cluster_labels == cluster_id)[0]
        centroid = embeddings[indices].mean(axis=0)
        closest_index = indices[
            np.argmin(np.linalg.norm(embeddings[indices] - centroid, axis=1))
        ]
        representatives.append((closest_index, embeddings[closest_index]))

    return representatives


def densest_doc_clusters(
    docs: List[Document], k: int, embedding_fn: Callable[[str], np.ndarray]
) -> List[Document]:
    """
    Find the documents corresponding to the representative vectors of the k densest
    clusters in the given list of documents.

    Args:
        docs (List[Document]): A list of Document instances, each containing a "content"
                               field to be embedded and a "metadata" field.
        k (int): Number of densest clusters to find.
        embedding_fn (Callable[[str], np.ndarray]): A function that maps a string to an
                                                    embedding vector.

    Returns:
        List[Document]: A list of Document instances corresponding to the representative
                        vectors of the k densest clusters.
    """

    # Extract embeddings from the documents
    embeddings = np.array(embedding_fn([doc.content for doc in docs]))

    # Find the densest clusters and their representative indices
    representative_indices_and_vectors = densest_clusters(embeddings, k)

    # Extract the corresponding documents
    representative_docs = [docs[i] for i, _ in representative_indices_and_vectors]

    return representative_docs
