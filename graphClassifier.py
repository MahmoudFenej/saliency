import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csgraph
import networkx as nx
from sklearn.cluster import DBSCAN
from lightgbm import LGBMClassifier
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import pyvista as pv
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from itertools import combinations
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to ModelNet10
MODELNET_PATH = "ModelNet10"

# Categories of the dataset
categories = [folder for folder in os.listdir(MODELNET_PATH) if os.path.isdir(os.path.join(MODELNET_PATH, folder))]

def load_modelnet_files(dataset_type='train', max_files=10):
    meshes = []
    labels = []
    file_count = 0
    
    logging.info(f"Loading {dataset_type} dataset...")

    for category in categories:
        category_dir = os.path.join(MODELNET_PATH, category, dataset_type)
        for filename in os.listdir(category_dir):
            if filename.endswith('.off'):
                mesh = trimesh.load(os.path.join(category_dir, filename))
                meshes.append(mesh)
                labels.append(category)
                file_count += 1
                logging.info(f"Loaded mesh: {filename}, Label: {category}")
            if file_count >= max_files:
                file_count = 0
                break
                
    logging.info(f"Loaded {file_count} meshes from {dataset_type} dataset.")
    return meshes, labels

def estimate_normals_and_eigenvalues_pca(vertices, neighbor_radii):
    normals = np.zeros(vertices.shape)
    eigenvalues_list = np.zeros((vertices.shape[0], 3))
    kdtree = cKDTree(vertices)
    for i, vertex in enumerate(vertices):
        radius = neighbor_radii[i] if neighbor_radii[i] > 0 else 0.02
        neighbors = kdtree.query_ball_point(vertex, r=radius)
        if len(neighbors) < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors])
        normals[i] = pca.components_[-1]
        eigenvalues_list[i] = pca.explained_variance_
    return normals, eigenvalues_list

def compute_local_covariance_features(eigenvalues):
    # Add safeguards to handle invalid values and zero divisions
    epsilon = 1e-6
    eigenvalues = np.maximum(eigenvalues, epsilon)  # Prevent division by zero
    
    # Calculate features
    total_eigenvalues = np.sum(eigenvalues, axis=1)
    linearity = eigenvalues[:, 0] / total_eigenvalues
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / total_eigenvalues
    sphericity = eigenvalues[:, 2] / total_eigenvalues
    omnivariance = np.cbrt(np.prod(eigenvalues, axis=1))
    anisotropy = (eigenvalues[:, 0] - eigenvalues[:, 1]) / (eigenvalues[:, 2] + epsilon)
    eigentropy = -np.sum(eigenvalues * np.log(eigenvalues + epsilon), axis=1)
    sum_eigenvalues = np.sum(eigenvalues, axis=1)
    local_surface_variation = np.mean(np.linalg.norm(np.diff(eigenvalues, axis=0), axis=1))
    
    # Check the shape of each feature array
    num_points = eigenvalues.shape[0]

    return np.hstack([
        linearity.reshape(-1, 1),
        planarity.reshape(-1, 1),
        sphericity.reshape(-1, 1),
        omnivariance.reshape(-1, 1),
        anisotropy.reshape(-1, 1),
        eigentropy.reshape(-1, 1),
        sum_eigenvalues.reshape(-1, 1),
        np.full((num_points, 1), local_surface_variation)  # Local Surface Variation is a single value
    ])


def compute_geometrical_features(vertices):
    kdtree = cKDTree(vertices)
    
    local_density = np.zeros(len(vertices))
    farthest_distance = np.zeros(len(vertices))
    max_height = np.zeros(len(vertices))
    height_std_dev = np.zeros(len(vertices))
    
    for i, vertex in enumerate(vertices):
        neighbors = kdtree.query_ball_point(vertex, 0.05)
        local_density[i] = len(neighbors)
        
        farthest_distance[i] = np.max(np.linalg.norm(vertices - vertex, axis=1))
        heights = vertices[:, 2]  # Assuming z-coordinate as height
        max_height[i] = np.max(heights)
        height_std_dev[i] = np.std(heights)
    
    return np.vstack([local_density, farthest_distance, max_height, height_std_dev]).T



def derive_novel_salient_features(vertices, saliency):
    salient_indices = np.where(saliency > 0.4)[0]
    if len(salient_indices) == 0:
        return np.zeros(5)
            
    perimeter, area  = compute_perimeter_area_and_graph(vertices, saliency, n_clusters=10, link_threshold=0.5, visualize=True)


    return perimeter, area


def compute_dynamic_neighbor_radii(vertices, k=10):
    kdtree = cKDTree(vertices)
    dists, _ = kdtree.query(vertices, k=k+1)
    avg_dist = np.mean(dists[:, 1:], axis=1)
    return avg_dist

def compute_dynamic_radius(mesh, vertices, scaling_factor=0.1):
    kdtree = cKDTree(vertices)
    dists, _ = kdtree.query(vertices, k=2)
    avg_dist = np.mean(dists[:, 1])
    bounding_box = mesh.bounding_box_oriented.extents
    bbox_diag = np.linalg.norm(bounding_box)
    r = scaling_factor * bbox_diag if bbox_diag > 0 else avg_dist * scaling_factor
    return r

def estimate_normals_and_eigenvalues_pca(vertices, neighbor_radii):
    normals = np.zeros(vertices.shape)
    eigenvalues_list = np.zeros((vertices.shape[0], 3))
    kdtree = cKDTree(vertices)
    for i, vertex in enumerate(vertices):
        radius = neighbor_radii[i] if neighbor_radii[i] > 0 else 0.02
        neighbors = kdtree.query_ball_point(vertex, r=radius)
        if len(neighbors) < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors])
        normals[i] = pca.components_[-1]  # The normal is the eigenvector associated with the smallest eigenvalue
        eigenvalues_list[i] = pca.explained_variance_
    return normals, eigenvalues_list


def estimate_curvature_nonparametric(vertices, normals, neighbor_radii):
    curvatures = np.zeros(vertices.shape[0])
    kdtree = cKDTree(vertices)
    for i, (vertex, normal) in enumerate(zip(vertices, normals)):
        radius = neighbor_radii[i] if neighbor_radii[i] > 0 else 0.02
        neighbors = kdtree.query_ball_point(vertex, r=radius)
        if len(neighbors) < 3:
            continue
        projections = [(normal.T @ (vertices[j] - vertex)) for j in neighbors]
        curvatures[i] = np.mean(projections)
    return curvatures

def compute_dk_dn(mesh, curvature, vertex_normals, dynamic_r, edge_boost=1.5):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])

    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=dynamic_r)
        if len(neighbors) < 1:
            continue
        
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        if curvature_diff.size > 0:
            dk_q[i] = np.mean(curvature_diff)
        
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        if normal_diff.size > 0:
            dn_q[i] = np.mean(normal_diff)

    return dk_q, dn_q

# Saliency computation
def compute_saliency(dk, dn):
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    return S_q

    
def derive_curvature_weighted_features(vertices, saliency, curvatures):
    salient_indices = np.where(saliency > 0.4)[0]
    if len(salient_indices) == 0:
        return np.zeros(2)
    
    salient_curvatures = curvatures[salient_indices]
    
    # Curvature-Based Saliency (average curvature of salient regions)
    curvature_based_saliency = np.mean(salient_curvatures)
    
    # Normalized Saliency Compactness
    salient_vertices = vertices[salient_indices]
    salient_center = np.mean(salient_vertices, axis=0)
    distances_to_center = np.linalg.norm(salient_vertices - salient_center, axis=1)
    compactness = np.mean(distances_to_center / (salient_curvatures + 1e-6))
    
    
    return np.array([curvature_based_saliency, compactness])

def compute_curvature_variance(vertices, curvatures, saliency):
    """ Compute Saliency-Aware Curvature Variance (SACV) for salient regions. """
    salient_indices = np.where(saliency > 0.4)[0]
    if len(salient_indices) == 0:
        return 0.0
    
    salient_curvatures = curvatures[salient_indices]
    curvature_variance = np.var(salient_curvatures)
    
    weighted_variance = np.sum(curvature_variance * saliency[salient_indices]) / np.sum(saliency[salient_indices])
    
    return weighted_variance



def compute_centroids_and_links(vertices, labels, salient_indices):
    unique_labels = set(labels)
    centroids = []

    for label in unique_labels:
        if label == -1:
            continue 

        cluster_points = vertices[salient_indices][labels == label]
        if len(cluster_points) == 0:
            continue
        
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)

    links = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            links.append([centroids[i], centroids[j]])

    return centroids, links

def compute_adjacency_matrix(centroids, links):
    num_centroids = len(centroids)
    adj_matrix = np.zeros((num_centroids, num_centroids))
    
    for link in links:
        idx1 = np.where(np.all(centroids == link[0], axis=1))[0][0]
        idx2 = np.where(np.all(centroids == link[1], axis=1))[0][0]
        adj_matrix[idx1, idx2] = 1
        adj_matrix[idx2, idx1] = 1 

    return adj_matrix

def compute_laplacian(adj_matrix):
    laplacian = csgraph.laplacian(adj_matrix, normed=True)
    return laplacian


def compute_perimeter_area_and_graph(vertices, saliency, n_clusters=5, link_threshold=0.5, visualize=False):
    salient_indices = np.where(saliency > 0.2)[0]
    if len(salient_indices) == 0:
        print("No salient points detected.")
        return np.zeros(5) 

    salient_vertices = vertices[salient_indices]
    
    # Adjust the number of clusters if there are fewer salient points than n_clusters
    adjusted_n_clusters = min(n_clusters, len(salient_vertices))
    
    clustering = AgglomerativeClustering(n_clusters=adjusted_n_clusters, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(salient_vertices)

    centroids, links = compute_centroids_and_links(vertices, labels, salient_indices)
        
    perimeter, area = calculate_graph_features(centroids, links)
   
    return perimeter, area

def visualize_clusters_with_links(vertices, labels, salient_indices, centroids, links):
    plotter = pv.Plotter()
    unique_labels = set(labels)

    # Loop through each cluster and add points
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points or unclustered data

        cluster_points = vertices[salient_indices][labels == label]
        random_color = np.random.rand(3)
        plotter.add_points(cluster_points, color=random_color, point_size=8)

    # Add centroids
    plotter.add_points(centroids, color="yellow", point_size=12, render_points_as_spheres=True)

    # Add links between centroids
    for link in links:
        plotter.add_lines(np.array([link[0], link[1]]), color="black", width=5)

    plotter.show()

def calculate_graph_features(centroids, links):
    perimeter = 0
    area = 0
    
    for link in links:
        perimeter += np.linalg.norm(link[0] - link[1])
    
    for p1, p2 in combinations(centroids, 2):
        for link in links:
            if np.array_equal(p1, link[0]) and np.array_equal(p2, link[1]):
                for p3 in centroids:
                    if np.array_equal(p3, p1) or np.array_equal(p3, p2):
                        continue
                    a = np.linalg.norm(p1 - p2)
                    b = np.linalg.norm(p2 - p3)
                    c = np.linalg.norm(p3 - p1)
                    s = (a + b + c) / 2
                    area += np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    return perimeter, area

def compress_with_histogram(descriptor, n_bins=3):
    """Compresses the descriptor by creating a histogram."""
    hist, bin_edges = np.histogram(descriptor, bins=n_bins)
    return hist

def summarize_hks(hks):
    return [
        np.mean(hks),
        np.std(hks),
        np.median(hks),
        np.max(hks),
        np.min(hks)
    ]

def extract_novel_features(mesh, mesh_index, total_meshes, label):
    vertices = mesh.vertices
    dynamic_r = compute_dynamic_radius(mesh, vertices)
    dynamic_neighbor_radii = compute_dynamic_neighbor_radii(vertices)

    normals_pca, eigenvalues_list = estimate_normals_and_eigenvalues_pca(vertices, dynamic_neighbor_radii)
    curvature = estimate_curvature_nonparametric(vertices, normals_pca, dynamic_neighbor_radii)
    dk, dn = compute_dk_dn(mesh, curvature, normals_pca, dynamic_r, edge_boost=1.5)
    saliency = compute_saliency(dk, dn)
    
    
    curvature_weighted_features = derive_curvature_weighted_features(vertices, saliency, curvature)
    sacv = compute_curvature_variance(vertices, curvature, saliency)


    local_covariance_features = compute_local_covariance_features(eigenvalues_list)
    geometrical_features = compute_geometrical_features(vertices)

    perimeter, area=  derive_novel_salient_features(vertices, saliency)

    features = np.hstack([
        np.mean(local_covariance_features, axis=0),
        np.mean(geometrical_features, axis=0),
        curvature_weighted_features,
        sacv,
        perimeter,
        area,
    ])
    
    logging.info(f"Features for {label} {mesh_index + 1}/{total_meshes}: {features}")
    # logging.info(f"Features for {label} {mesh_index + 1}/{total_meshes}")
    
    return features

def process_meshes(meshes, labels):
    feature_matrix = []
    for index, mesh in enumerate(meshes):
        features = extract_novel_features(mesh, index, len(meshes), labels[index])
        feature_matrix.append(features)
    return np.array(feature_matrix)

def train_and_evaluate_models(train_features, train_labels, test_features, test_labels):
    logging.info("Training and evaluating models...")
    
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)
    
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=10)
    model.fit(scaled_train_features, train_labels)

    predictions = model.predict(scaled_test_features)
    accuracy = accuracy_score(test_labels, predictions)
    logging.info(f"Accuracy with Random forest: {accuracy :.4f}")



if __name__ == "__main__":

    logging.info("Extracting features from training meshes...")
    train_meshes, train_labels = load_modelnet_files('train', max_files=100)
    train_features = process_meshes(train_meshes, train_labels)
    
    logging.info("Extracting features from test meshes...")
    test_meshes, test_labels = load_modelnet_files('test', max_files=20)
    test_features = process_meshes(test_meshes, test_labels)
    
    # Check if features are valid
    if train_features.size == 0 or test_features.size == 0:
        logging.error("Feature extraction failed. Please check the data and try again.")
    else:
        train_and_evaluate_models(train_features, train_labels, test_features, test_labels)
