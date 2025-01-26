import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to ModelNet10
MODELNET_PATH = "ModelNet10"

# Categories of the dataset
categories = [folder for folder in os.listdir(MODELNET_PATH) if os.path.isdir(os.path.join(MODELNET_PATH, folder))]

def load_modelnet_files(dataset_type='train', max_files=100):
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
                break
                
    logging.info(f"Loaded {file_count} meshes from {dataset_type} dataset.")
    return meshes, labels

# Load train and test data
train_meshes, train_labels = load_modelnet_files(dataset_type='train', max_files=100)
test_meshes, test_labels = load_modelnet_files(dataset_type='test', max_files=50)

def compute_nonparam_curvature_pca(vertices, neighbor_radii):
    curvatures = np.zeros(vertices.shape[0])
    kdtree = cKDTree(vertices)
    for i, vertex in enumerate(vertices):
        radius = neighbor_radii[i] if neighbor_radii[i] > 0 else 0.02  # Ensure radius is not zero
        neighbors = kdtree.query_ball_point(vertex, r=radius)
        if len(neighbors) < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors])
        eigenvalues = pca.explained_variance_
        if np.sum(eigenvalues) == 0:
            curvatures[i] = 0  # Avoid division by zero
        else:
            curvatures[i] = eigenvalues[-1] / np.sum(eigenvalues)  # Smallest eigenvalue for curvature
    return curvatures

def compute_dynamic_neighbor_radii(vertices, k=10):
    kdtree = cKDTree(vertices)
    dists, _ = kdtree.query(vertices, k=k+1)  # k+1 to include the point itself
    avg_dist = np.mean(dists[:, 1:], axis=1)  # Compute average distance excluding the point itself
    return avg_dist

def compute_dynamic_radius(mesh, vertices, scaling_factor=0.1):
    kdtree = cKDTree(vertices)
    dists, _ = kdtree.query(vertices, k=2)
    avg_dist = np.mean(dists[:, 1])
    bounding_box = mesh.bounding_box_oriented.extents
    bbox_diag = np.linalg.norm(bounding_box)
    r = scaling_factor * bbox_diag if bbox_diag > 0 else avg_dist * scaling_factor
    return r

def compute_dk_dn(mesh, curvature, vertex_normals, dynamic_r):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=dynamic_r)
        if len(neighbors) < 1:
            continue
        
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.mean(curvature_diff)
        
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.mean(normal_diff)
        
    return dk_q, dn_q

def compute_saliency(dk, dn):
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    return S_q

def extract_features_from_mesh(mesh):
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    
    dynamic_r = compute_dynamic_radius(mesh, vertices)
    dynamic_neighbor_radii = compute_dynamic_neighbor_radii(vertices)
    
    curvature = compute_nonparam_curvature_pca(vertices, dynamic_neighbor_radii)
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals, dynamic_r)
    saliency = compute_saliency(dk, dn)
    
    salient_indices = np.where(saliency > 0)[0]
    salient_points = vertices[salient_indices]
    
    logging.info(f"Saliency computed, {len(salient_points)} salient points identified.")
    
    G = nx.Graph()
    for i, point in enumerate(salient_points):
        G.add_node(i, pos=tuple(point))
    
    if salient_points.size > 0:
        kdtree = cKDTree(salient_points)
        for i, point in enumerate(salient_points):
            neighbors = kdtree.query(point, k=3)[1]  # Get indices of k-nearest neighbors
            for neighbor in neighbors:
                if neighbor != i:
                    G.add_edge(i, neighbor)
    
    degrees = np.array([G.degree(node) for node in G.nodes])
    clustering = np.array([nx.clustering(G, node) for node in G.nodes])
    betweenness = np.array([nx.betweenness_centrality(G, normalized=True)[node] for node in G.nodes])
    eigenvector = np.array([nx.eigenvector_centrality(G, max_iter=1000)[node] for node in G.nodes])
    
    if len(G.nodes) > 0:
        features = np.array([np.mean(degrees), np.mean(clustering), np.mean(betweenness), np.mean(eigenvector)])
        logging.info(f"Features extracted: {features}")
    else:
        features = np.zeros(4)  # Return zero features if no salient points are found
        logging.info("No salient points found, returning zero features.")
    
    return features

# Extract features for all meshes
train_features = np.array([extract_features_from_mesh(mesh) for mesh in train_meshes])
test_features = np.array([extract_features_from_mesh(mesh) for mesh in test_meshes])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
test_labels_enc = le.transform(test_labels)

# Train a Random Forest Classifier
logging.info("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_features, train_labels_enc)

# Predict on the test set
test_predictions = clf.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels_enc, test_predictions)
logging.info(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
