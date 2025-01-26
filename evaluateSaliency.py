import os
import numpy as np
import trimesh
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numpy.linalg import eig
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.linalg import eigh
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import networkx as nx

# Function to load meshes and ground truths
def load_mesh_and_ground_truth(mesh_path, ground_truth_path):
    mesh = trimesh.load(mesh_path)
    with open(ground_truth_path, 'r') as file:
        ground_truth = np.array([float(line.strip()) for line in file])
    return mesh, ground_truth

def compute_weight(v1, v2, weighting, params=None):
    dist = np.linalg.norm(v1 - v2)
    params = params or {}

    if weighting == "gaussian":
        distance = np.linalg.norm(v1 - v2)
        sigma = params.get("sigma", 1.0)
        return np.exp(-distance**2 / (2 * sigma**2))

    elif weighting == "inverse_distance":
        p = params.get("p", 2)
        epsilon = params.get("epsilon", 1e-8)
        return 1 / (dist**p + epsilon)

    elif weighting == "binary":
        r = params.get("radius", 0.05)
        return 1 if dist <= r else 0

    elif weighting == "cosine_similarity":
        vec = v2 - v1
        norm = np.linalg.norm(vec)
        return vec.dot(vec) / (norm**2 + 1e-8)

    elif weighting == "angle_based":
        vec = v2 - v1
        angle = np.arccos(np.clip(vec[0] / np.linalg.norm(vec), -1, 1))
        return np.cos(angle)

    elif weighting == "adaptive_gaussian":
        sigma = params.get("sigma", dist)  # Use distance as adaptive sigma
        return np.exp(-dist**2 / (2 * sigma**2))

    elif weighting == "entropy":
        p = dist / (params.get("scale", dist + 1e-8))
        return -p * np.log(p + 1e-8)

    elif weighting == "curvature_based":
        curvature_i = params.get("curvature_i", 0)
        curvature_j = params.get("curvature_j", 0)
        return abs(curvature_i - curvature_j)

    else:
        raise ValueError(f"Unknown weighting type: {weighting}")
    


def build_mesh_graph(vertices, faces, weighting="gaussian", params=None):
 
    graph = nx.Graph()

    graph.add_nodes_from(range(len(vertices)))

    for face in faces:
        for i in range(3):
            v1_idx, v2_idx = face[i], face[(i + 1) % 3]
            v1, v2 = vertices[v1_idx], vertices[v2_idx]
            weight = compute_weight(v1, v2, weighting, params)
            if weight > 0: 
                graph.add_edge(v1_idx, v2_idx, weight=weight)

    return graph


def estimate_normals_pca_graph(vertices, graph, scales=[0.001, 0.002, 0.01]):

    normals = np.zeros(vertices.shape)
    eigenvalues_list = np.zeros((vertices.shape[0], 3))

    for i in range(len(vertices)):
        combined_neighbors = set()
        for radius in scales:
            neighbors = [
                j for j in graph.neighbors(i)
                if np.linalg.norm(vertices[i] - vertices[j]) <= radius
            ]
            combined_neighbors.update(neighbors)

        combined_neighbors = list(combined_neighbors)
        
        if len(combined_neighbors) < 3:
            continue
        
        pca = PCA(n_components=3)
        pca.fit(vertices[combined_neighbors])
        normals[i] = pca.components_[-1]
        eigenvalues_list[i] = pca.explained_variance_

    return normals, eigenvalues_list



def estimate_curvature_multi_scale(vertices, normals, graph):
    curvatures = np.zeros((vertices.shape[0]))
    for i, (vertex, normal) in enumerate(zip(vertices, normals)):
        if i not in graph:
            continue

        neighbors = [n for n in graph.neighbors(i) if graph[i][n]['weight']] if i in graph else []

        if len(neighbors) < 3:
            continue

        projections = [(normal.T @ (vertices[j] - vertex)) for j in neighbors]
        curvatures[i] = np.mean(projections)

    return curvatures


# Compute dk and dn with graph-based neighborhood
def compute_dk_dn_graph(graph, vertices, curvature, vertex_normals):
    dn_q = np.zeros(vertices.shape[0])
    dk_q = np.zeros(vertices.shape[0])
    for i in range(len(vertices)):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) < 3:
            continue
        weights = np.array([graph[i][j]['weight'] for j in neighbors])
        
        normal_diffs = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.sum(normal_diffs * weights) / np.sum(weights)
        
        curvature_diffs = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.sum(curvature_diffs * weights) / np.sum(weights)
        
    return dk_q, dn_q

# Saliency computation
def compute_saliency(dn_q, dk_q):
    S_q = 2 - (np.exp(-dn_q) + np.exp(-dk_q))
    return S_q


def binarize_saliency(saliency_array, threshold=0.5):
    # Convert saliency values to 0 (non-salient) or 1 (salient)
    binary_mask = np.where(saliency_array > threshold, 1, 0)
    return binary_mask

# Define paths
mesh_dir = "SchellingData/SchellingData/Meshes/"
ground_truth_dir = "SchellingData/SchellingData/Distributions/"

# Get all mesh files
mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.off')]

# Initialize lists for precision and recall
precisions = []
recalls = []

mesh_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.off')],
                    key=lambda x: int(x.split('.')[0]))  # Sort numerically by the number before .off

for mesh_file in mesh_files:
    # Load the mesh and ground truth
    mesh_path = os.path.join(mesh_dir, mesh_file)
    ground_truth_file = mesh_file.replace('.off', '.val')
    ground_truth_path = os.path.join(ground_truth_dir, ground_truth_file)
    
    # Check if ground truth file exists
    if not os.path.exists(ground_truth_path):
        print(f"Ground truth file not found for {mesh_file}. Skipping...")
        continue

    mesh, ground_truth = load_mesh_and_ground_truth(mesh_path, ground_truth_path)
    
    vertices = mesh.vertices
    faces = mesh.faces

    params = {"sigma": 0.02, "p": 2, "radius": 0.05}

    graph = build_mesh_graph(vertices, faces, weighting="gaussian", params=params)
    normals_pca, eigenvalues_list = estimate_normals_pca_graph(vertices, graph)
    curvature = estimate_curvature_multi_scale(vertices, normals_pca, graph)
    dk, dn = compute_dk_dn_graph(graph, vertices, curvature, normals_pca)
    saliency = compute_saliency(dk, dn)

    # Binarize saliency and ground truth
    predicted_mask = binarize_saliency(saliency, threshold=0)
    ground_truth_mask = binarize_saliency(ground_truth, threshold=0)

    # Flatten the masks for precision and recall
    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    # Compute Precision and Recall
    precision = precision_score(ground_truth_mask_flat, predicted_mask_flat)
    recall = recall_score(ground_truth_mask_flat, predicted_mask_flat)
    
    if(precision > 0 and recall > 0):
        print(f"Mesh: {mesh_file}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        precisions.append(precision)
        recalls.append(recall)

average_precision = np.mean(precisions)
average_recall = np.mean(recalls)

print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")

