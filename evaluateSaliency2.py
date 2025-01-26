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

# Function to load meshes and ground truths
def load_mesh_and_ground_truth(mesh_path, ground_truth_path):
    mesh = trimesh.load(mesh_path)
    with open(ground_truth_path, 'r') as file:
        ground_truth = np.array([float(line.strip()) for line in file])
    return mesh, ground_truth

def estimate_curvature_nonparametric(vertices, normals, faces):
    num_vertices = vertices.shape[0]
    curvatures = np.zeros(num_vertices)

    # Create a dictionary to hold neighbors for each vertex based on faces
    neighbors_dict = {i: set() for i in range(num_vertices)}

    # Populate the neighbors dictionary using faces
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors_dict[face[i]].add(face[j])

    # Convert sets to lists for easier indexing later
    for key in neighbors_dict:
        neighbors_dict[key] = list(neighbors_dict[key])

    # Compute curvature for each vertex
    for i, normal in enumerate(normals):
        neighbors = neighbors_dict[i]
        if len(neighbors) < 3:
            continue

        # Collect neighbors' positions and compute projections
        projections = [np.dot(normal, vertices[j] - vertices[i]) for j in neighbors]

        # The curvature is the mean of the projections
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

def estimate_normals_using_covariance(mesh):
    num_vertices = mesh.vertices.shape[0]
    estimated_normals = np.zeros((num_vertices, 3))

    # Create a dictionary to hold neighbors for each vertex
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

    # Iterate over each vertex
    for i in range(num_vertices):
        neighbors = neighbors_dict[i]
        if len(neighbors) < 3:
            continue  # Need at least 3 points to compute normal

        # Collect the coordinates of the vertex and its neighbors
        points = mesh.vertices[neighbors]

        # Step 1: Compute centroid of the neighborhood
        centroid = np.mean(points, axis=0)

        # Step 2: Compute covariance matrix of the neighborhood
        centered_points = points - centroid
        covariance_matrix = np.dot(centered_points.T, centered_points) / len(points)

        # Step 3: Perform eigenvalue decomposition of the covariance matrix
        eigenvalues, eigenvectors = eigh(covariance_matrix)

        # Step 4: The normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]  # Corresponds to smallest eigenvalue

        # Store the normal
        estimated_normals[i] = normal

    return estimated_normals

def compute_saliency(dk, dn):
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    # S_q[S_q < 0.23] = 0
    return S_q


def multi_scale_weighted_differences_dynamic(mesh, curvature, normals, scales=[1, 2, 3]):
    num_vertices = mesh.vertices.shape[0]
    dk_q = np.zeros(num_vertices)
    dn_q = np.zeros(num_vertices)
    
    neighbors_dict = {i: set() for i in range(num_vertices)}
    for face in mesh.faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors_dict[face[i]].add(face[j])

    for i in range(num_vertices):
        neighbors = list(neighbors_dict[i])
        if len(neighbors) < 1:
            continue

        q_point = mesh.vertices[i]
        neighbor_points = mesh.vertices[neighbors]

        # Dynamically adjust scales based on local curvature
        local_curvature = curvature[i]
        scale_factor = np.clip(local_curvature, 0.1, 1.0)  # Clip to avoid extreme values

        for scale in scales:
            adjusted_scale = scale * scale_factor
            distances = np.linalg.norm(neighbor_points - q_point, axis=1)
            weights = np.exp(-(distances / adjusted_scale)**2)

            area_element = np.sum(weights)

            curvature_diff = np.abs(curvature[i] - curvature[neighbors])
            dk_q[i] += np.sum(curvature_diff * weights) / area_element

            normal_diff = np.linalg.norm(normals[i] - normals[neighbors], axis=1)
            dn_q[i] += np.sum(normal_diff * weights) / area_element

        dk_q[i] /= len(scales)
        dn_q[i] /= len(scales)

    return dk_q, dn_q


def normalize_features(features):
    min_val = np.min(features)
    max_val = np.max(features)
    return (features - min_val) / (max_val - min_val)


def compute_local_geometric_saliency(dk, dn, mesh, radius=0.01):
    """
    Compute saliency based on local geometric features (curvature and normal differences).

    Parameters:
    - dk: Curvature differences for each vertex.
    - dn: Normal differences for each vertex.
    - mesh: The 3D mesh to evaluate.
    - radius: Radius for local neighborhood to compute features.

    Returns:
    - S_q: Saliency values for each vertex.
    """
    num_vertices = mesh.vertices.shape[0]
    saliency = np.zeros(num_vertices)

    # Compute global means and standard deviations
    global_dk_mean = np.mean(dk)
    global_dk_std = np.std(dk)
    global_dn_mean = np.mean(dn)
    global_dn_std = np.std(dn)

    # Create a KD-Tree for efficient neighborhood search
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)

    for i in range(num_vertices):
        # Find neighbors within a specified radius
        neighbors_idx = tree.query_ball_point(mesh.vertices[i], radius)
        
        # Skip if no neighbors found
        if len(neighbors_idx) < 2:
            continue
        
        # Local curvature and normal differences
        local_dk = dk[neighbors_idx]
        local_dn = dn[neighbors_idx]

        # Calculate local mean and std
        local_dk_mean = np.mean(local_dk)
        local_dk_std = np.std(local_dk)
        local_dn_mean = np.mean(local_dn)
        local_dn_std = np.std(local_dn)

        # Calculate z-scores for dk and dn
        dk_z = (dk[i] - local_dk_mean) / (local_dk_std + 1e-6)  # Avoid division by zero
        dn_z = (dn[i] - local_dn_mean) / (local_dn_std + 1e-6)  # Avoid division by zero

        # Compute saliency as a combination of z-scores
        saliency[i] = np.abs(dk_z) + np.abs(dn_z)  # Use absolute values to emphasize changes

    # Normalize the saliency values
    saliency = normalize_features(saliency)
    
    return saliency

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
    
    vertex_normals = estimate_normals_using_covariance(mesh)
    curvature = estimate_curvature_nonparametric(mesh.vertices, vertex_normals, faces)
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
    saliency = compute_saliency(dk, dn)

    # Compute multi-scale curvature and normal differences
    dk, dn = multi_scale_weighted_differences_dynamic(mesh, curvature, vertex_normals)

    # Compute enhanced saliency
    saliency = compute_local_geometric_saliency(dk, dn, mesh)
        
    # mean_gt = np.mean(ground_truth)
    # std_gt = np.std(ground_truth)
    # mean_saliency = np.mean(saliency)
    # std_saliency = np.std(saliency)
    # saliency_transformed = (saliency - mean_saliency) * (std_gt / std_saliency) + mean_gt
    # saliency_transformed = np.clip(saliency_transformed, 0, 1)

    # Binarize saliency and ground truth
    predicted_mask = binarize_saliency(saliency, threshold=0)
    ground_truth_mask = binarize_saliency(ground_truth, threshold=0)

    # Flatten the masks for precision and recall
    predicted_mask_flat = predicted_mask.flatten()
    ground_truth_mask_flat = ground_truth_mask.flatten()

    # Compute Precision and Recall
    precision = precision_score(ground_truth_mask_flat, predicted_mask_flat)
    recall = recall_score(ground_truth_mask_flat, predicted_mask_flat)
    
    # Print individual mesh precision and recall

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
