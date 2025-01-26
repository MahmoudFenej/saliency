import os
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphSAGE
import pyvista as pv
import re
from sklearn.model_selection import train_test_split
import logging
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from torch_geometric.nn import MessagePassing
from sklearn.metrics import precision_score, recall_score
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_scatter import scatter_add
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the weight computation function
def compute_weight(v1, v2, weighting="gaussian", params=None):
    dist = np.linalg.norm(v1 - v2)
    params = params or {}

    if weighting == "gaussian":
        sigma = params.get("sigma", 0.02)
        return np.exp(-dist**2 / (2 * sigma**2))
    elif weighting == "inverse_distance":
        p = params.get("p", 2)
        epsilon = params.get("epsilon", 1e-8)
        return 1 / (dist**p + epsilon)
    elif weighting == "binary":
        r = params.get("radius", 0.1)
        return 1 if dist <= r else 0
    else:
        raise ValueError(f"Unknown weighting type: {weighting}")

# Estimate normals using PCA
def estimate_normals_pca(vertices, graph, scales=[0.001, 0.005, 0.01]):
    normals = np.zeros(vertices.shape)
    eigenvalues_list = np.zeros((vertices.shape[0], 3))

    for i in range(len(vertices)):
        neighbors = []
        for radius in scales:
            neighbors += [
                j for j in graph.neighbors(i)
                if np.linalg.norm(vertices[i] - vertices[j]) <= radius
            ]
        neighbors = list(set(neighbors))
        
        if len(neighbors) < 3:
            continue
        
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors])
        normals[i] = pca.components_[-1]
        eigenvalues_list[i] = pca.explained_variance_

    return normals, eigenvalues_list

# Estimate curvature for multiple scales
def estimate_curvature(vertices, normals, graph):
    curvatures = np.zeros(vertices.shape[0])
    for i, normal in enumerate(normals):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) < 3:
            continue

        projections = [
            np.dot(normals[j], (vertices[j] - vertices[i]))
            for j in neighbors
        ]
        curvatures[i] = np.var(projections)

    return curvatures


# Build the updated mesh graph with new features
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


# Compute curvature sign (Convex/Concave)
def compute_curvature_sign(curvatures):
    return np.sign(curvatures)


# Compute normal deviation (angular)
def compute_normal_deviation(normals, graph):
    deviations = np.zeros(len(normals))
    for i in range(len(normals)):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) < 2:
            continue
        normal_devs = np.array([np.dot(normals[i], normals[n]) for n in neighbors])
        deviations[i] = np.mean(np.abs(normal_devs - 1))  # Mean angular deviation
    return deviations


# Normalize vertex positions
def normalize_vertex_positions(vertices):
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    return (vertices - min_vals) / (max_vals - min_vals)


def compute_principal_curvatures(normals, vertices, graph):
    k1 = np.zeros(len(vertices))
    k2 = np.zeros(len(vertices))
    
    for i, normal in enumerate(normals):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) < 3:
            continue
        
        neighbors_coords = vertices[neighbors] - vertices[i]
        proj = neighbors_coords - np.outer(np.dot(neighbors_coords, normal), normal)
        
        pca = PCA(n_components=2)
        pca.fit(proj)
        
        # Principal curvatures (approximation based on eigenvalues of the projected covariance)
        eigenvalues = pca.explained_variance_
        if len(eigenvalues) >= 2:
            k1[i], k2[i] = eigenvalues[0], eigenvalues[1]
    
    return k1, k2

# Compute Shape Index
def compute_shape_index(k1, k2):
    shape_index = np.zeros(len(k1))
    for i in range(len(k1)):
        if k1[i] != k2[i]:
            shape_index[i] = (2 / np.pi) * np.arctan((k1[i] + k2[i]) / (k1[i] - k2[i]))
    return shape_index

# Compute Gaussian Curvature
def compute_gaussian_curvature(k1, k2):
    return k1 * k2

# Compute Curvature Index
def compute_curvature_index(k1, k2):
    return (np.abs(k1) + np.abs(k2)) / 2

# Update Feature Extraction to Use the New Features
def compute_geometric_features_updated(mesh, graph):
    vertices = mesh.vertices
    
    # Compute normals using PCA
    normals, _ = estimate_normals_pca(vertices, graph)
    
    # Compute principal curvatures
    k1, k2 = compute_principal_curvatures(normals, vertices, graph)
    
    # Compute shape index, Gaussian curvature, and curvature index
    shape_index = compute_shape_index(k1, k2)
    gaussian_curvature = compute_gaussian_curvature(k1, k2)
    curvature_index = compute_curvature_index(k1, k2)
    
    # Combine features into a matrix
    feature_matrix = np.stack([shape_index, gaussian_curvature, curvature_index], axis=1)
    
    return feature_matrix


def compute_saliency(dn_q, dk_q):
    S_q = 2 - (np.exp(-dn_q) + np.exp(-dk_q))
    return S_q

# Updated mesh data loading function
def load_mesh_data(mesh_dir, ground_truth_dir, max_meshes=20):
    dataset = []
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.off')]
    mesh_files = sorted(mesh_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    if max_meshes:
        mesh_files = mesh_files[:max_meshes]

    for mesh_file in mesh_files:
        logger.info(f"Processing mesh: {mesh_file}")
        mesh_path = os.path.join(mesh_dir, mesh_file)
        ground_truth_path = os.path.join(ground_truth_dir, mesh_file.replace('.off', '.val'))

        if not os.path.exists(ground_truth_path):
            logger.warning(f"Ground truth file missing for {mesh_file}, skipping...")
            continue

        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices
        faces = mesh.faces

        params = {"sigma": 0.02, "p": 2, "radius": 0.05}
        graph = build_mesh_graph(vertices, faces, weighting="gaussian", params=params)
        
        normals_pca, eigenvalues_list = estimate_normals_pca_graph(vertices, graph)
        curvature = estimate_curvature_multi_scale(vertices, normals_pca, graph)
        dk, dn = compute_dk_dn_graph(graph, vertices, curvature, normals_pca)
        # saliency_scores = compute_saliency(dk, dn)

        features = compute_geometric_features_updated(mesh, graph)

        # features = np.hstack([features, 
        # ])

        # Create PyTorch Geometric Data
        x = torch.tensor(features, dtype=torch.float)
        edges = list(graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor([graph[u][v]['weight'] for u, v in edges], dtype=torch.float)

        ground_truth = torch.tensor(np.loadtxt(ground_truth_path), dtype=torch.long)
        ground_truth_normalized = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=ground_truth_normalized, mesh_path=mesh_path, ground_truth_path=ground_truth_path)
        dataset.append(data)

    return dataset

class BasicGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BasicGNNLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops to the adjacency matrix
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))

        # Normalize edge weights
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        edge_weight = edge_weight / (deg[row] + 1e-8)

        # Message Passing
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.linear(aggr_out)

# Define the GNN Model
class SaliencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SaliencyGNN, self).__init__()
        self.gnn1 = GraphSAGE(input_dim, hidden_dim, num_layers=3)
        self.gnn2 = GraphSAGE(hidden_dim, hidden_dim, num_layers=3)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gnn1(x, edge_index, edge_weight))
        x = F.relu(self.gnn2(x, edge_index, edge_weight))
        return self.linear(x)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # Add the first layer (from input_dim to hidden_dim)
        self.convs.append(GraphSAGE(input_dim, hidden_dim, aggr='mean', num_layers=num_layers))
        
        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already have the input layer and output layer
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, aggr='mean', num_layers=num_layers))

        # Add the final output layer (hidden_dim to output_dim)
        self.convs.append(GraphSAGE(hidden_dim, output_dim, aggr='mean', num_layers=num_layers))

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Apply GraphSAGE layers
        for i in range(self.num_layers - 1):  # We process up to the second-to-last layer with ReLU
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.dropout(x)

        # The last layer (without ReLU)
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x.squeeze()
        
# Visualize a mesh
def visualize_mesh_pyvista(vertices, faces, saliency=None):
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)
    if saliency is not None:
        mesh.point_data['Saliency'] = saliency
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='Saliency', cmap='Reds')
    plotter.show() 


    
# Main script for regression
if __name__ == "__main__":
    mesh_dir = "SchellingData/SchellingData/Meshes/"
    ground_truth_dir = "SchellingData/SchellingData/Distributions/"

    dataset = load_mesh_data(mesh_dir, ground_truth_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    input_dim = 3
    hidden_dim = 64
    output_dim = 1  # Single continuous value for regression
    model = GNN(input_dim, hidden_dim, output_dim, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
            total_loss = 0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                predictions = torch.sigmoid(model(batch))
                loss = criterion(predictions, batch.y)
                # loss = node_losses.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            mesh = trimesh.load(data.mesh_path[0])
            saliency = torch.sigmoid(model(data.x, data.edge_index)).squeeze().numpy()
            # saliency = np.where(saliency < 0.5, 0, saliency)
            with open(data.ground_truth_path[0], 'r') as file:
                ground_truth = np.array([float(line.strip()) for line in file])

            # mean_gt = np.mean(ground_truth)
            # std_gt = np.std(ground_truth)
            # mean_saliency = np.mean(saliency)
            # std_saliency = np.std(saliency)
            # saliency_transformed = (saliency - mean_saliency) * (std_gt / std_saliency) + mean_gt
            # saliency_transformed = np.clip(saliency_transformed, 0, 1)
            # ground_truth_normalized = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)

            predicted_mask = np.where(saliency > 0, 1, 0)
            ground_truth_mask = np.where(ground_truth > 0, 1, 0)
            precision = precision_score(ground_truth_mask, predicted_mask)
            recall = recall_score(ground_truth_mask, predicted_mask)
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            
            print(saliency)

            np.savetxt("saliency_values.txt", saliency)
            visualize_mesh_pyvista(mesh.vertices, mesh.faces, saliency=saliency)