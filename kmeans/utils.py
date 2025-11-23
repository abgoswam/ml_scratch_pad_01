import matplotlib.pyplot as plt
import numpy as np

def plot_kmeans(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    centroids: np.ndarray,
    label_clusters: bool = True,
    draw_centroids: bool = True,
) -> None:
    assert X.ndim == 2 and X.shape[1] == 2
    assert labels.shape == (X.shape[0],)
    assert labels.min() >= 0 and labels.max() < n_clusters
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    if label_clusters:
        # Plot each cluster
        for cluster_id in range(n_clusters):
            plt.scatter(
                X[labels == cluster_id, 0],
                X[labels == cluster_id, 1],
                label=f'Cluster {cluster_id}'
            )
    else:
        plt.scatter(X[:, 0], X[:, 1])
    
    if draw_centroids:
        # Plot centroids
        plt.scatter(
            centroids[:, 0], centroids[:, 1],
            c='orange', s=100, marker='X',
            linewidths=.2,
            alpha=0.5,
            label='Centroids'
        )
    
    
    plt.title('KMeans Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metric(K_range: np.ndarray, metrics: np.ndarray) -> None:
    assert K_range.ndim == 1 and K_range.shape == metrics.shape
    # Plot the elbow curve
    plt.figure(figsize=(8, 4))
    plt.plot(K_range, metrics, 'bo-', markersize=6)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Metric')
    plt.title('Metric / number of clusters.')
    plt.grid(True)
    plt.show()


def patch_kmeans_with_defaults(cls):
    def metric_naive(kmeans_model, points: np.ndarray) -> float:
        # Returns sum of distances from points to corresponding centroids.
        assignments = kmeans_model.get_assignments(points)
        return np.sum((points - kmeans_model.centroids[assignments, :]) ** 2)
    
    if not hasattr(cls, "metric"):
        setattr(cls, "metric", lambda self, points: metric_naive(self, points))

    obj = cls(2)
    obj._initialize(np.array([[1, 2], [3, 4]]))
    try:
        obj.get_assignments(np.array([[1, 2], [3, 4]]))
    except NotImplementedError:
        # If the method throws NotImplementedError, add default implementation of returning random cluster assignments.
        setattr(cls, "get_assignments", lambda self, points: np.random.randint(0, self.n_clusters, size=(points.shape[0],)))
    
    try:
        obj._recalculate_centroids(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    except NotImplementedError:
        # If the method throws NotImplementedError, add default implementation of returning random points.
        setattr(cls, "_recalculate_centroids", lambda self, points, new_assignments: points[np.random.choice(len(points), self.n_clusters, replace=False), :])