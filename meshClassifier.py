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

def get_class_id(mesh_index):
    """Map mesh index to class ID."""
    if 1 <= mesh_index <= 20:
        return 0  # Human -> class 0
    elif 21 <= mesh_index <= 40:
        return 1  # Cup -> class 1
    elif 41 <= mesh_index <= 60:
        return 2  # Glasses -> class 2
    elif 61 <= mesh_index <= 80:
        return 3  # Airplane -> class 3
    elif 81 <= mesh_index <= 100:
        return 4  # Ant -> class 4
    elif 101 <= mesh_index <= 120:
        return 5  # Chair -> class 5
    elif 121 <= mesh_index <= 140:
        return 6  # Octopus -> class 6
    elif 141 <= mesh_index <= 160:
        return 7  # Table -> class 7
    elif 161 <= mesh_index <= 180:
        return 8  # Teddy -> class 8
    elif 181 <= mesh_index <= 200:
        return 9  # Hand -> class 9
    elif 201 <= mesh_index <= 220:
        return 10 # Plier -> class 10
    elif 221 <= mesh_index <= 240:
        return 11 # Fish -> class 11
    elif 241 <= mesh_index <= 260:
        return 12 # Bird -> class 12
    elif 281 <= mesh_index <= 300:
        return 13 # Armadillo -> class 13
    elif 301 <= mesh_index <= 320:
        return 14 # Bust -> class 14
    elif 321 <= mesh_index <= 340:
        return 15 # Mech -> class 15
    elif 341 <= mesh_index <= 360:
        return 16 # Bearing -> class 16
    elif 361 <= mesh_index <= 380:
        return 17 # Vase -> class 17
    elif 381 <= mesh_index <= 400:
        return 18 # Fourleg -> class 18
    return None

# Updated mesh data loading function (without ground truth labels)
import os
import re
import trimesh
from torch_geometric.data import Data
from collections import defaultdict

def load_mesh_data(mesh_dir, max_meshes=180):
    dataset = []
    class_labels = set()  # To store unique class labels
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.off')]
    mesh_files = sorted(mesh_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    if max_meshes:
        mesh_files = mesh_files[:max_meshes]
    
    class_meshes = defaultdict(list)
    for mesh_file in mesh_files:
        mesh_index = int(re.search(r'(\d+)', mesh_file).group())
        label = get_class_id(mesh_index)
        class_labels.add(label)

        # Check if label is valid (not None)
        if label is None:
            logger.error(f"Label is None for mesh {mesh_file} with index {mesh_index}")
            continue  # Skip this mesh if the label is None

        class_meshes[label].append(mesh_file)

    # For each class, take up to 'max_meshes' meshes
    for class_id, mesh_files in class_meshes.items():
        for mesh_file in mesh_files:  # Iterate over the mesh files for each class
            logger.info(f"Processing mesh: {mesh_file}")
            mesh_path = os.path.join(mesh_dir, mesh_file)

            try:
                mesh = trimesh.load(mesh_path)
                vertices = mesh.vertices
                faces = mesh.faces

                # Get label from class
                label = class_id

                params = {"sigma": 0.02, "p": 2, "radius": 0.05}
                graph = build_mesh_graph(vertices, faces, weighting="gaussian", params=params)

                normals_pca, eigenvalues_list = estimate_normals_pca_graph(vertices, graph)
                curvature = estimate_curvature_multi_scale(vertices, normals_pca, graph)
                dk, dn = compute_dk_dn_graph(graph, vertices, curvature, normals_pca)

                features = compute_geometric_features_updated(mesh, graph)

                x = torch.tensor(features, dtype=torch.float)
                edges = list(graph.edges())
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_weights = torch.tensor([graph[u][v]['weight'] for u, v in edges], dtype=torch.float)

                label_tensor = torch.tensor(label, dtype=torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=label_tensor, mesh_path=mesh_path)
                dataset.append(data)

            except Exception as e:
                logger.error(f"Failed to load or process mesh {mesh_file}: {e}")
                continue  # Skip this mesh if any error occurs

    return dataset, len(class_labels)  # Return the dataset and the number of unique classes

# Define the GNN Model
class MeshClassificationGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MeshClassificationGNN, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # Add the first layer (from input_dim to hidden_dim)
        self.convs.append(GraphSAGE(input_dim, hidden_dim, aggr='mean', num_layers=num_layers))
        
        # Add hidden layers
        for _ in range(num_layers - 2):  # For intermediate layers
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

        # The last layer (without ReLU) to get the output logits for classification
        x = self.convs[-1](x, edge_index, edge_weight)
        
        # Aggregate over all vertices for mesh-level prediction (mean or sum)
        x = x.mean(dim=0)  # Aggregate the per-vertex predictions to get a mesh-level prediction
        
        return x  # Shape will be (num_classes,)
    

# Main script for classification
if __name__ == "__main__":
    mesh_dir = "SchellingData/SchellingData/Meshes/"

    dataset, num_classes = load_mesh_data(mesh_dir)
    print(num_classes)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    input_dim = 3
    hidden_dim = 64
    output_dim = num_classes  # Number of classes (based on the `get_class_id` function)
    model = MeshClassificationGNN(input_dim, hidden_dim, output_dim, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
from sklearn.metrics import accuracy_score, precision_score, recall_score

for epoch in range(20):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)

        # Ensure the output shape is (1, num_classes) for batch size 1
        output = output.unsqueeze(0)

        target = data.y  # target is already a 1D tensor of shape (1,)

        loss = F.cross_entropy(output, target)  # Cross-entropy loss between output and target

        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate the loss for logging

    # Output the average training loss for this epoch
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}")
    
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for data in val_loader:
        output = model(data)

        output = output.unsqueeze(0)  # Add batch dimension

        preds = output.argmax(dim=-1)  # Get the class with the highest probability
        all_preds.append(preds.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
            
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

print(f"Epoch {epoch}, Evaluation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
