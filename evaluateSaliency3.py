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
import datetime

# Function to load meshes and ground truths
def load_mesh_and_ground_truth(mesh_path, ground_truth_path):
    mesh = trimesh.load(mesh_path)
    with open(ground_truth_path, 'r') as file:
        ground_truth = np.array([float(line.strip()) for line in file])
    return mesh, ground_truth

def estimate_curvature_nonparametric(vertices, normals, neighbors_dict):
    num_vertices = vertices.shape[0]
    curvatures = np.zeros(num_vertices)

    for i, normal in enumerate(normals):
        neighbors = neighbors_dict[i]
        if len(neighbors) < 3:
            continue

        projections = [np.dot(normal, vertices[j] - vertices[i]) for j in neighbors]
        curvatures[i] = np.mean(projections)

    return curvatures


# Function to compute differences in curvature and normal for each vertex
def compute_dk_dn(mesh, curvature, vertex_normals):
    num_vertices = mesh.vertices.shape[0]
    dk_q = np.zeros(num_vertices)
    dn_q = np.zeros(num_vertices)

    # Create a dictionary to hold neighbors for each vertex based on mesh faces
    neighbors_dict = {i: set() for i in range(num_vertices)}

    # Populate the neighbors dictionary using faces
    for face in mesh.faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors_dict[face[i]].add(face[j])

    # Convert sets to lists for easier indexing later
    for key in neighbors_dict:
        neighbors_dict[key] = list(neighbors_dict[key])

    # Iterate over each vertex q
    for i in range(num_vertices):
        neighbors = neighbors_dict[i]
        if len(neighbors) < 1:
            continue
        
        # Get coordinates of current vertex q and its neighbors
        q_point = mesh.vertices[i]
        neighbor_points = mesh.vertices[neighbors]

        # Compute distances from q to each neighbor
        distances = np.linalg.norm(neighbor_points - q_point, axis=1)

        # Gaussian weighting function (decaying with distance)
        weights = np.exp(-distances**2 / 2)  # Can adjust the scale if needed

        # Sum of weights (approximation of surface element dxdy)
        area_element = np.sum(weights)

        # Compute weighted curvature difference for dk(q)
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.sum(curvature_diff * weights) / area_element  # Weighted average of curvature differences

        # Compute weighted normal difference for dn(q)
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.sum(normal_diff * weights) / area_element  # Weighted average of normal differences

    return dk_q, dn_q

def estimate_normals_using_covariance(neighbors_dict):
    normals = np.zeros(vertices.shape)

    for i in range(vertices.shape[0]):
        neighbors = neighbors_dict[i]
        if len(neighbors) < 3:
            continue
        
        cov_matrix = np.cov(vertices[neighbors].T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        normals[i] = eigenvectors[:, 0]

    return normals


def compute_neighbors_kdtree(vertices):
    tree = KDTree(vertices)
    neighbors = {i: tree.query_ball_point(vertices[i], 0.01) for i in range(len(vertices))}
    return neighbors

def normalize_features(features):
    min_val = np.min(features)
    max_val = np.max(features)
    return (features - min_val) / (max_val - min_val)


def binarize_saliency(saliency_array, threshold=0.5):
    # Convert saliency values to 0 (non-salient) or 1 (salient)
    binary_mask = np.where(saliency_array > threshold, 1, 0)
    return binary_mask

def compute_saliency(dk_values, dn_values, curvature_weight=0.2):
   
    start = datetime.datetime.now()
    print("Computing saliency... ")

    # Calculate min and max for dk and dn
    min_k = min(dk_values)
    max_k = max(dk_values)
    min_n = min(dn_values)
    max_n = max(dn_values)

    diff_k = max_k - min_k
    diff_n = max_n - min_n
    normal_weight = 1 - curvature_weight
    print('diff_k {}, diff_n {}'.format(diff_k, diff_n))

    # Normalize dk and dn
    normed_dk = [(dk - min_k) / diff_k for dk in dk_values]
    normed_dn = [(dn - min_n) / diff_n for dn in dn_values]

    # Compute saliency for each point
    saliency_values = [
        normal_weight * norm_dn + curvature_weight * norm_dk
        for norm_dn, norm_dk in zip(normed_dn, normed_dk)
    ]

    diff = datetime.datetime.now() - start
    print("Done. Processing took %.2f [s]" % diff.total_seconds())

    return saliency_values

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
    
    neighbors = compute_neighbors_kdtree(vertices)

    vertex_normals = estimate_normals_using_covariance(neighbors)

    curvature = estimate_curvature_nonparametric(mesh.vertices, vertex_normals, neighbors)

    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)

    saliency = compute_saliency(dk, dn)
    
    saliency = np.array(saliency)

    predicted_mask = binarize_saliency(saliency, threshold=0)
    ground_truth_mask = binarize_saliency(ground_truth, threshold=0)

    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    precision = precision_score(ground_truth_mask_flat, predicted_mask_flat)
    recall = recall_score(ground_truth_mask_flat, predicted_mask_flat)
    
    if(precision > 0 and recall > 0):
        print(f"Mesh: {mesh_file}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        precisions.append(precision)
        recalls.append(recall)

# Calculate average precision and recall
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)

# Print average precision and recall
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")

# Optionally, visualize last processed mesh and its saliency
# visualize_mesh_pyvista(vertices, faces, saliency_transformed)
