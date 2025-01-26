import numpy as np
import trimesh
from sklearn.decomposition import PCA
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import networkx as nx
from scipy.spatial import Delaunay

# Load the mesh
mesh = trimesh.load("SchellingData/SchellingData/Meshes/30.off")
ground_truth_path = "SchellingData/SchellingData/Distributions/30.val"
with open(ground_truth_path, 'r') as file:
    ground_truth = np.array([float(line.strip()) for line in file])

vertices = mesh.vertices
faces = mesh.faces

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

# Visualization using PyVista
def visualize_mesh_pyvista(vertices, faces, saliency=None):
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)
    if saliency is not None:
        mesh.point_data['Saliency'] = saliency
    plotter = pv.Plotter(window_size=[1000, 600])
    cmap = 'Reds' if saliency is not None else 'viridis'
    plotter.add_mesh(mesh, scalars='Saliency' if saliency is not None else None, cmap=cmap)
    plotter.show()


params = {"sigma": 0.02, "p": 2, "radius": 0.05}

weighting_methods = [
    "gaussian", "inverse_distance", "binary",
    "cosine_similarity", "angle_based",
    "adaptive_gaussian", "entropy"
]

graph = build_mesh_graph(vertices, faces, weighting="gaussian", params=params)
normals_pca, eigenvalues_list = estimate_normals_pca_graph(vertices, graph)
curvature = estimate_curvature_multi_scale(vertices, normals_pca, graph)
dk, dn = compute_dk_dn_graph(graph, vertices, curvature, normals_pca)

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
