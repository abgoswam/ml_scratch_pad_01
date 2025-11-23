import numpy as np


def test_get_assignments_1(cls) -> None:
    n_clusters = 2
    points = np.array([[0, 0], [1, 1]])
    centroids = np.array([[0,0], [1,1]])
    expected_assignments = np.array([0, 1])
    
    print(f"Points: {points}")
    print(f"Centroids: {centroids}")
    print(f"Expected assignments: {expected_assignments}")

    obj = cls(n_clusters=n_clusters)
    obj.centroids = centroids
    assignments = obj.get_assignments(points)

    print(f"Actual assignments: {assignments}")

    assert assignments.shape == (points.shape[0],)
    assert assignments.dtype == int
    assert np.all(assignments == expected_assignments)
    print("SUCCESS")


def test_get_assignments_2(cls) -> None:
    n_clusters = 2
    points = np.array([[0, 0], [1, 1], [.5, .6], [.5, .4], [.8, 0], [.8, .8]])
    centroids = np.array([[0, 0], [1, 1]])
    expected_assignments = np.array([0, 1, 1, 0, 0, 1])

    print(f"Points: {points}")
    print(f"Centroids: {centroids}")
    print(f"Expected assignments: {expected_assignments}")

    obj = cls(n_clusters=n_clusters)
    obj.centroids = centroids
    assignments = obj.get_assignments(points)

    print(f"Actual assignments: {assignments}")

    assert assignments.shape == (points.shape[0],)
    assert assignments.dtype == int
    assert np.all(assignments == expected_assignments)
    print("SUCCESS")


def test_recalculate_centroids_1(cls) -> None:
    n_clusters = 2
    points = np.array([[0, 0], [1, 1]])
    assignments = np.array([0, 1])
    expected_centroids = np.array([[0,0], [1,1]])
    
    print(f"Points: {points}")
    print(f"Assignments: {assignments}")
    print(f"Expected centroids: {expected_centroids}")

    obj = cls(n_clusters=n_clusters)
    obj.centroids = np.zeros((n_clusters, points.shape[1]))
    centroids = obj._recalculate_centroids(points, assignments)
    
    print(f"Actual centroids: {centroids}")

    assert centroids.shape == (n_clusters, points.shape[1])
    assert np.allclose(centroids, expected_centroids)
    print("SUCCESS")


def test_recalculate_centroids_2(cls) -> None:
    n_clusters = 2
    points = np.array([[0, 0], [1, 1], [.5, .6], [.5, .4], [.8, 0], [.8, .8]])
    assignments = np.array([0, 1, 1, 0, 0, 1])
    expected_centroids = np.array([
        [1.3 / 3, .4 / 3],
        [2.3 / 3, 2.4 / 3]
    ])

    print(f"Points: {points}")
    print(f"Assignments: {assignments}")
    print(f"Expected centroids: {expected_centroids}")

    obj = cls(n_clusters=n_clusters)
    obj.centroids = np.zeros((n_clusters, points.shape[1]))
    centroids = obj._recalculate_centroids(points, assignments)
    
    print(f"Actual centroids: {centroids}")

    assert centroids.shape == (n_clusters, points.shape[1])
    assert np.allclose(centroids, expected_centroids)
    print("SUCCESS")
