import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

np.random.seed(42)


def test_sample_1(seed: int = 42) -> tuple[int, np.ndarray]:
    np.random.seed(seed)
    points: list[list[float]] = [
        [ 5,  3],
        [ 6,  3],
        [ 4,  3],
        
        [-6, -3],
        [-5, -3],
        [-4, -3],
        
        [-6,  3],
        [-5,  3],
        [-4,  3],
        
        [ 5, -3],
        [ 4, -3],
        [ 6, -3],
    ]
    n_clusters = 4
    points_arr = np.array(points)
    np.random.shuffle(points_arr)
    return n_clusters, points_arr


def test_sample_2_3_helper(seed: int, scale: float) -> tuple[int, np.ndarray]:
    np.random.seed(seed)
    points = np.random.uniform(low=scale, high=scale * 1.2, size=(20, 2))
    n_clusters = 2
    return n_clusters, points

def test_sample_2(seed: int = 42) -> tuple[int, np.ndarray]:
    return test_sample_2_3_helper(seed, scale=1e7)

def test_sample_3(seed: int = 42) -> tuple[int, np.ndarray]:
    return test_sample_2_3_helper(seed, scale=1e-7)

def test_sample_4(seed: int = 42) -> tuple[int, np.ndarray]:
    # Create 3 clusters along x1 axis, noise in high-scale x2
    np.random.seed(42)
    n = 300
    
    # Actual clustering happens along x1
    x1 = np.hstack([
        np.random.normal(loc=-5, scale=0.5, size=n),
        np.random.normal(loc=0, scale=0.5, size=n),
        np.random.normal(loc=5, scale=0.5, size=n)
    ])
    
    # x2 is meaningless high-range noise
    x2 = np.random.uniform(0, 10000, size=3 * n)
    
    X = np.vstack((x1, x2)).T

    return 3, X

def test_sample_5(seed: int = 42) -> tuple[int, np.ndarray]:
    points, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    return 2, points, y_true

def test_sample_6(seed: int = 42) -> tuple[int, np.ndarray]:
    X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=0)
    return 2, X, y_true

def test_sample_7(seed: int = 42) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(seed)

    def generate_ray_cluster(angle_deg, angle_width, n=100, r_min=1.0, r_max=10.0, noise=0.1):
        angle_rad = np.deg2rad(rng.uniform(angle_deg, angle_deg+angle_width))
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        radii = rng.uniform(r_min, r_max, size=n)
        base_points = direction * radii[:, None]
        jitter = rng.normal(scale=noise, size=(n, 2))
        return base_points + jitter

    # Use rays that are closer in angle
    points = np.vstack([
        generate_ray_cluster(3, 5, r_max=15.0),
        generate_ray_cluster(15, 5, r_max=15.0),
        generate_ray_cluster(25, 5, r_max=15.0),
    ])

    return 3, points


def test_sample_8(seed: int = 42) -> np.ndarray:
    np.random.seed(seed)

    X, _ = make_blobs(
        n_samples=1000,
        centers=7,
        cluster_std=[0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.6],
        center_box=(-5, 5),
        random_state=42
    )

    return X


def test_sample_9(seed: int = 42) -> tuple[int, np.ndarray]:
    # Create 6 evenly spaced clusters in a line
    centers = [[i * 5, 0] for i in range(6)]  # Spaced apart to seem mergeable at small K
    X, _ = make_blobs(n_samples=1200, centers=centers, cluster_std=0.8, random_state=42)
    return X


def test_sample_10(seed: int = 42, tol: float = 1e-4) -> np.ndarray:
    np.random.seed(seed)

    # Set cluster_std based on tol: make it ~10× smaller so convergence takes fine updates
    cluster_std = tol * 0.001

    # Set center box so clusters are within about ±10× tol from origin
    range_scale = tol * 10
    center_box = (-range_scale, range_scale)

    # Generate the data
    n_clusters = 3
    samples_per_cluster = 100
    total_samples = samples_per_cluster * n_clusters
    X, _ = make_blobs(
        n_samples=total_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=seed
    )

    return n_clusters, X

def test_sample_11(seed: int = 42) -> tuple[int, np.ndarray]:
    np.random.seed(seed)

    # Cluster A & B are close, Cluster C is far
    centers = [[0, 0], [0.5, 0.5], [10, 10]]  # A and B will likely merge
    X_real, _ = make_blobs(n_samples=[100, 100, 100], centers=centers, cluster_std=0.3, random_state=seed)
    
    # Add one far outlier
    outlier = np.array([[100, 100]])
    X = np.vstack([X_real, outlier])

    return 3, X

def test_sample_12(seed: int = 42) -> tuple[int, np.ndarray]:
    np.random.seed(seed)

    # === Parameters ===
    n_samples = 1000
    n_clusters = 3
    signal_dims = 5
    noise_dims = 9995
    total_dims = signal_dims + noise_dims
    
    # === Step 1: Generate signal ===
    X_signal, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=signal_dims,
        cluster_std=1.0,
        random_state=seed
    )
    
    # === Step 2: Add high-dimensional noise ===
    X_noise = np.random.normal(loc=0, scale=5.0, size=(n_samples, noise_dims))
    X_full = np.hstack([X_signal, X_noise])

    return n_clusters, X_full


def test_sample_13(seed: int = 42) -> tuple[int, np.ndarray]:
    np.random.seed(seed)
    
    # Create 2 clusters: one tight, one spread out
    X1, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.5, random_state=seed+1)
    X2, _ = make_blobs(n_samples=300, centers=[[5, 0]], cluster_std=2.5, random_state=seed+2)
    
    X = np.vstack([X1, X2])

    return 2, X


def test_sample_14(seed: int = 42) -> tuple[int, np.ndarray]:
    np.random.seed(seed)
    X, _ = make_blobs(n_samples=30, centers=3, cluster_std=1.0, random_state=seed)
    return 3, X
