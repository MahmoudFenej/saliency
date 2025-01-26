import numpy as np
import trimesh
from sklearn.decomposition import PCA
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Load the mesh
mesh = trimesh.load("SchellingData/SchellingData/Meshes/15.off")
ground_truth_path = "SchellingData/SchellingData/Distributions/15.val"
with open(ground_truth_path, 'r') as file:
    ground_truth = np.array([float(line.strip()) for line in file])

vertices = mesh.vertices
faces = mesh.faces

# Adaptive normal estimation using PCA with neighborhood adjustment
def estimate_normals_pca_adaptive(vertices, min_neighbors=10, max_neighbors=50):
    normals = np.zeros(vertices.shape)
    eigenvalues_list = np.zeros((vertices.shape[0], 3))
    kdtree = cKDTree(vertices)
    for i, vertex in enumerate(vertices):
        neighbors = kdtree.query(vertex, k=max_neighbors)[1]
        if len(neighbors) < min_neighbors:
            continue
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors])
        normals[i] = pca.components_[-1]
        eigenvalues_list[i] = pca.explained_variance_
    return normals, eigenvalues_list

# Multi-scale curvature estimation using PCA-based projections
def estimate_curvature_multi_scale(vertices, normals, scales=(0.01, 0.02, 0.03)):
    curvatures = np.zeros((vertices.shape[0], len(scales)))
    kdtree = cKDTree(vertices)
    for scale_idx, radius in enumerate(scales):
        for i, (vertex, normal) in enumerate(zip(vertices, normals)):
            neighbors = kdtree.query_ball_point(vertex, r=radius)
            if len(neighbors) < 3:
                continue
            projections = [(normal.T @ (vertices[j] - vertex)) for j in neighbors]
            curvatures[i, scale_idx] = np.mean(projections)
    # Combine multi-scale curvatures by averaging
    return np.mean(curvatures, axis=1)

# Improved compute_dk_dn function with Gaussian weighting
def compute_dk_dn_adaptive(mesh, curvature, vertex_normals, sigma=0.01, radius=0.02):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    kdtree = cKDTree(mesh.vertices)
    
    vertex_area = np.full(mesh.vertices.shape[0], 1.0)
    
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=radius)
        if len(neighbors) < 10:
            continue
        neighbor_vertices = mesh.vertices[neighbors]
        distances = np.linalg.norm(neighbor_vertices - q, axis=1)
        w = np.exp(-distances**2 / (2 * sigma**2))
        area_weights = vertex_area[neighbors] * w

        normal_diffs = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.sum(normal_diffs * area_weights) / np.sum(area_weights)

        curvature_diffs = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.sum(curvature_diffs * area_weights) / np.sum(area_weights)
        
    return dk_q, dn_q

# Saliency computation
def compute_saliency(dn_q, dk_q):
    S_q = 2 - (np.exp(-dn_q) + np.exp(-dk_q))
    return S_q

# Visualization using PyVista
def visualize_mesh_pyvista(vertices, faces, saliency=None):
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)
    if saliency is not None:
        mesh.point_data['Saliency'] = saliency
    plotter = pv.Plotter(window_size=[1000, 600])
    cmap = 'Reds' if saliency is not None else 'viridis'
    if saliency is not None:
        plotter.add_mesh(mesh, scalars='Saliency', cmap=cmap)
    else:
        plotter.add_mesh(mesh, color='viridis')
    plotter.show()

# Execute the computation
normals_pca, eigenvalues_list = estimate_normals_pca_adaptive(vertices)
curvature = estimate_curvature_multi_scale(vertices, normals_pca)
dk, dn = compute_dk_dn_adaptive(mesh, curvature, normals_pca)

# Compute saliency
saliency = compute_saliency(dk, dn)

# Scale and transform saliency for visualization
mean_gt = np.mean(ground_truth)
std_gt = np.std(ground_truth)
mean_saliency = np.mean(saliency)
std_saliency = np.std(saliency)
saliency_transformed = (saliency - mean_saliency) * (std_gt / std_saliency) + mean_gt
saliency_transformed = np.clip(saliency_transformed, 0, 1)

# Print and visualize results
print("Saliency values:\n", saliency)
visualize_mesh_pyvista(vertices, faces, saliency_transformed)
visualize_mesh_pyvista(vertices, faces, ground_truth)

# Plot histograms for comparison
plt.hist(saliency_transformed, bins=50, alpha=0.5, label='Saliency')
plt.hist(ground_truth, bins=50, alpha=0.5, label='Ground Truth')
plt.legend()
plt.show()

# Precision and Recall Computation
predicted_mask = np.where(saliency > 0, 1, 0)
ground_truth_mask = np.where(ground_truth > 0, 1, 0)
precision = precision_score(ground_truth_mask, predicted_mask)
recall = recall_score(ground_truth_mask, predicted_mask)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
