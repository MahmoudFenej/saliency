import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import networkx as nx

# Define directories for ModelNet dataset
MODELNET_PATH = "ModelNet10"  # Update this to your path
categories = [folder for folder in os.listdir(MODELNET_PATH) if os.path.isdir(os.path.join(MODELNET_PATH, folder))]

def load_modelnet_files(dataset_type='train', max_files=1000):
    meshes = []
    labels = []
    file_count = 0  # Counter for the number of files loaded
    
    for category in categories:
        category_dir = os.path.join(MODELNET_PATH, category, dataset_type)
        for filename in os.listdir(category_dir):
            if filename.endswith('.off'):
                mesh = trimesh.load(os.path.join(category_dir, filename))
                labels.append(category)
                meshes.append(mesh)
                file_count += 1
                
    return meshes, labels

def compute_nonparam_curvature(mesh):
    L = trimesh.smoothing.laplacian_calculation(mesh)
    curvature = L.dot(mesh.vertices)
    curvature_magnitude = np.linalg.norm(curvature, axis=1)
    return curvature_magnitude

def compute_dk_dn(mesh, curvature, vertex_normals):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=0.7)
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

def compute_local_covariance_features(mesh, salient_indices):
    kdtree = cKDTree(mesh.vertices)
    covariance_features = []
    
    for idx in salient_indices:
        neighbors = kdtree.query_ball_point(mesh.vertices[idx], r=0.7)
        if len(neighbors) < 3:
            covariance_features.append([0] * 8)
            continue
        
        local_points = mesh.vertices[neighbors] - mesh.vertices[idx]
        covariance_matrix = np.cov(local_points.T)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        sphericity = eigenvalues[2] / eigenvalues[0]
        omnivariance = (np.prod(eigenvalues)) ** (1 / 3)
        anisotropy = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
        eigentropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
        sum_of_eigenvalues = np.sum(eigenvalues)
        local_surface_variation = eigenvalues[2] / sum_of_eigenvalues
        
        covariance_features.append([linearity, planarity, sphericity, omnivariance, anisotropy, eigentropy, sum_of_eigenvalues, local_surface_variation])
    
    return np.array(covariance_features)

def compute_geometrical_features(mesh, salient_indices):
    kdtree = cKDTree(mesh.vertices)
    geometrical_features = []
    
    for idx in salient_indices:
        neighbors = kdtree.query_ball_point(mesh.vertices[idx], r=0.7)
        if len(neighbors) < 1:
            geometrical_features.append([0] * 4)
            continue
        
        local_points = mesh.vertices[neighbors]
        distances = np.linalg.norm(local_points - mesh.vertices[idx], axis=1)
        
        local_density = len(neighbors)
        farthest_distance = np.max(distances)
        max_height = np.max(local_points[:, 2])
        height_std = np.std(local_points[:, 2])
        
        geometrical_features.append([local_density, farthest_distance, max_height, height_std])
    
    return np.array(geometrical_features)

def compute_ssd_features(mesh, salient_indices):
    kdtree = cKDTree(mesh.vertices)
    ssd_features = []
    
    for idx in salient_indices:
        neighbors = kdtree.query_ball_point(mesh.vertices[idx], r=0.7)
        if len(neighbors) < 4:
            ssd_features.append([0] * 3)
            continue
        
        selected_points = mesh.vertices[np.random.choice(neighbors, 4, replace=False)]
        
        centroid = np.mean(selected_points, axis=0)
        dist_to_centroid = np.mean(np.linalg.norm(selected_points - centroid, axis=1))
        dist_between_two = np.linalg.norm(selected_points[0] - selected_points[1])
        sqrt_area = np.sqrt(np.linalg.norm(np.cross(selected_points[1] - selected_points[0], selected_points[2] - selected_points[0])) / 2)
        cubic_root_volume = np.abs(np.dot(selected_points[3] - selected_points[0], np.cross(selected_points[1] - selected_points[0], selected_points[2] - selected_points[0]))) ** (1 / 3)
        
        ssd_features.append([dist_to_centroid, dist_between_two, sqrt_area, cubic_root_volume])
    
    return np.array(ssd_features)

def extract_features(mesh, saliency_threshold=0.8):
    vertex_normals = mesh.vertex_normals
    curvature = compute_nonparam_curvature(mesh)
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
    saliency = compute_saliency(dk, dn)
    
    salient_indices = np.where(saliency > saliency_threshold)[0]
    if len(salient_indices) == 0:
        salient_indices = np.arange(len(mesh.vertices))
    
    salient_curvature = curvature[salient_indices]
    
    covariance_features = compute_local_covariance_features(mesh, salient_indices)
    geometrical_features = compute_geometrical_features(mesh, salient_indices)
    ssd_features = compute_ssd_features(mesh, salient_indices)
    
    curvature_summary = [np.mean(salient_curvature), np.std(salient_curvature), skew(salient_curvature), kurtosis(salient_curvature)]
    saliency_summary = [np.mean(saliency[salient_indices]), np.std(saliency[salient_indices]), skew(saliency[salient_indices]), kurtosis(saliency[salient_indices])]
    
    covariance_summary = np.mean(covariance_features, axis=0)
    geometrical_summary = np.mean(geometrical_features, axis=0)
    ssd_summary = np.mean(ssd_features, axis=0)
    
    features = np.concatenate([curvature_summary, saliency_summary, covariance_summary, geometrical_summary, ssd_summary])
    
    return features

def process_meshes(meshes, saliency_threshold=0.8):
    feature_list = []
    for idx, mesh in enumerate(meshes):
        features = extract_features(mesh, saliency_threshold)
        feature_list.append(features)
        print(f"Feature extraction done for mesh {idx + 1}/{len(meshes)}")
    features_array = np.array(feature_list)
    print(f"Features shape: {features_array.shape}")
    print(f"Features type: {features_array.dtype}")
    return features_array

# Load and process train and test sets
train_meshes, train_labels = load_modelnet_files('train')
test_meshes, test_labels = load_modelnet_files('test')

print("-----------------train loaded------------------")
print(train_meshes, train_labels)

# Extract features
train_features = process_meshes(train_meshes)
test_features = process_meshes(test_meshes)

# Label encoding
label_to_int = {label: idx for idx, label in enumerate(categories)}
train_labels_encoded = np.array([label_to_int[label] for label in train_labels])
test_labels_encoded = np.array([label_to_int[label] for label in test_labels])

# Standardize features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
train_features_scaled = imputer.fit_transform(train_features_scaled)
test_features_scaled = imputer.transform(test_features_scaled)

# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features_scaled, train_labels_encoded)

# Test the model
test_predictions = rf_model.predict(test_features_scaled)
accuracy = accuracy_score(test_labels_encoded, test_predictions)

# Output results
print("Accuracy on test set: {:.2f}%".format(accuracy * 100))
