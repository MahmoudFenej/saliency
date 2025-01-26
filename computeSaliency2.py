import numpy as np
import trimesh
from sklearn.decomposition import PCA
import pyvista as pv
from scipy.spatial import Delaunay
from sklearn.metrics import precision_score, recall_score
from numpy.linalg import eig
import matplotlib.pyplot as plt

# Load the mesh
mesh = trimesh.load("SchellingData/SchellingData/Meshes/1.off")
ground_truth_path = "SchellingData/SchellingData/Distributions/1.val"
with open(ground_truth_path, 'r') as file:
    ground_truth = np.array([float(line.strip()) for line in file])

vertices = mesh.vertices
faces = mesh.faces

# Compute neighbors using Delaunay triangulation
def compute_neighbors_delaunay(vertices):
    delaunay = Delaunay(vertices)
    neighbors = {i: set() for i in range(len(vertices))}
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                neighbors[simplex[i]].add(simplex[j])
                neighbors[simplex[j]].add(simplex[i])
    return neighbors

# Compute adaptive neighborhood based on local curvature variations
def compute_adaptive_neighbors(vertices, k=8):
    neighbors = {}
    for i in range(len(vertices)):
        distances = np.linalg.norm(vertices - vertices[i], axis=1)
        nearest_indices = np.argsort(distances)[1:k+1]  # Skip the vertex itself
        neighbors[i] = set(nearest_indices)
    return neighbors

# Compute normals based on PCA for local neighborhoods
def compute_normal(vertices, neighbors):
    normals = []
    for i in range(len(vertices)):
        neighbor_indices = list(neighbors[i])
        neighbor_vertices = vertices[neighbor_indices]
        pca = PCA(n_components=3)
        pca.fit(neighbor_vertices)
        normal = pca.components_[-1]  # The last component corresponds to the smallest variance direction
        normals.append(normal)
    return np.array(normals)

# Compute principal curvatures using PCA eigenvalues
def compute_principal_curvatures(vertices, neighbors):
    k1 = np.zeros(vertices.shape[0])
    k2 = np.zeros(vertices.shape[0])
    
    for i in range(len(vertices)):
        neighbor_indices = neighbors[i]
        if len(neighbor_indices) < 3:
            continue
        neighbor_vertices = vertices[list(neighbor_indices)] - vertices[i]
        covariance_matrix = np.cov(neighbor_vertices.T)
        eigenvalues, _ = eig(covariance_matrix)
        sorted_eigenvalues = np.sort(eigenvalues)
        k1[i], k2[i] = sorted_eigenvalues[-2:]  # Principal curvatures
    return k1, k2

# Compute Gaussian curvature using principal curvatures
def compute_gaussian_curvature(k1, k2):
    return k1 * k2

# Compute mean curvature using principal curvatures
def compute_mean_curvature(k1, k2):
    return 0.5 * (k1 + k2)

# Compute saliency based on adaptive curvature features
def compute_saliency(mean_curvature, gaussian_curvature, k1, k2, alpha=0.6, beta=0.4):
    mean_curvature_normalized = (mean_curvature - np.min(mean_curvature)) / (np.max(mean_curvature) - np.min(mean_curvature))
    gaussian_curvature_normalized = (gaussian_curvature - np.min(gaussian_curvature)) / (np.max(gaussian_curvature) - np.min(gaussian_curvature))
    
    # Adaptive weights based on local curvature variations
    k1_norm = (k1 - np.min(k1)) / (np.max(k1) - np.min(k1))
    k2_norm = (k2 - np.min(k2)) / (np.max(k2) - np.min(k2))
    weights = k1_norm + k2_norm  # Increase weight for regions with higher curvature
    
    saliency = (alpha * mean_curvature_normalized + beta * np.abs(gaussian_curvature_normalized)) * weights
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))  # Normalize
    return saliency

# Visualize the mesh with saliency
def visualize_mesh_pyvista(vertices, faces, saliency=None):
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)
    if saliency is not None:
        mesh.point_data['Saliency'] = saliency
    plotter = pv.Plotter(window_size=[1000, 600])
    cmap = 'Reds' if saliency is not None else 'viridis'
    plotter.add_mesh(mesh, scalars='Saliency' if saliency is not None else None, cmap=cmap)
    plotter.show()

# Main processing
neighbors = compute_adaptive_neighbors(vertices)
normals = compute_normal(vertices, neighbors)
k1, k2 = compute_principal_curvatures(vertices, neighbors)
mean_curvature = compute_mean_curvature(k1, k2)
gaussian_curvature = compute_gaussian_curvature(k1, k2)
saliency = compute_saliency(mean_curvature, gaussian_curvature, k1, k2)

# Shift and scale saliency to match ground truth distribution
mean_gt = np.mean(ground_truth)
std_gt = np.std(ground_truth)
mean_saliency = np.mean(saliency)
std_saliency = np.std(saliency)
saliency_transformed = (saliency - mean_saliency) * (std_gt / std_saliency) + mean_gt
saliency_transformed = np.clip(saliency_transformed, 0, 1)

# Visualization
visualize_mesh_pyvista(vertices, faces, saliency_transformed)
visualize_mesh_pyvista(vertices, faces, ground_truth)

# Plot histogram for comparison
plt.hist(saliency_transformed, bins=50, alpha=0.5, label='Saliency')
plt.hist(ground_truth, bins=50, alpha=0.5, label='Ground Truth')
plt.legend()
plt.show()

# Binarize saliency for comparison
def binarize_saliency(saliency_array, threshold=0):
    return np.where(saliency_array > threshold, 1, 0)

predicted_mask = binarize_saliency(saliency_transformed, 0)
ground_truth_mask = binarize_saliency(ground_truth, threshold=0)

# Compute Precision and Recall
precision = precision_score(ground_truth_mask.flatten(), predicted_mask.flatten())
recall = recall_score(ground_truth_mask.flatten(), predicted_mask.flatten())

print(f"Precision: {precision}")
print(f"Recall: {recall}")
