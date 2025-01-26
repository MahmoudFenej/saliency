import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
import networkx as nx
import trimesh
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("gnn_training.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Compute weight function
def compute_weight(v1, v2, weighting, params=None):
    dist = np.linalg.norm(v1 - v2)
    params = params or {}

    if weighting == "gaussian":
        sigma = params.get("sigma", 1.0)
        return np.exp(-dist**2 / (2 * sigma**2))

    elif weighting == "inverse_distance":
        p = params.get("p", 2)
        epsilon = params.get("epsilon", 1e-8)
        return 1 / (dist**p + epsilon)

    elif weighting == "binary":
        r = params.get("radius", 0.05)
        return 1 if dist <= r else 0

    else:
        raise ValueError(f"Unknown weighting type: {weighting}")


# Build graph for the mesh
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


# Estimate normals using PCA
def estimate_normals_pca_graph(vertices, graph, scales=[0.001, 0.002, 0.01]):
    normals = np.zeros(vertices.shape)
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
    return normals


# Compute curvature
def estimate_curvature_multi_scale(vertices, normals, graph):
    curvatures = np.zeros(len(vertices))
    for i, normal in enumerate(normals):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) < 3:
            continue
        projections = [normal.T @ (vertices[j] - vertices[i]) for j in neighbors]
        curvatures[i] = np.mean(np.abs(projections))
    return curvatures


# Compute dk and dn
def compute_dk_dn_graph(graph, vertices, curvature, vertex_normals):
    dn_q = np.zeros(len(vertices))
    dk_q = np.zeros(len(vertices))
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


# Create graph for GNN
def create_graph_for_gnn(vertices, faces, dk, dn):
    graph = build_mesh_graph(vertices, faces)
    edges = np.array(graph.edges).T
    edge_weights = np.array([graph[u][v]['weight'] for u, v in graph.edges])
    node_features = np.hstack([vertices, dk.reshape(-1, 1), dn.reshape(-1, 1)])
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float),
    )


# GNN Model
class SaliencyGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SaliencyGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Training loop with logging
def train_gnn(model, data_loader, optimizer, criterion, epochs=50):
    writer = SummaryWriter(log_dir="runs/saliency_gnn")
    model.train()
    logger.info("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    writer.close()


# Training loop with logging
def train_gnn(model, data_loader, optimizer, criterion, epochs=50):
    writer = SummaryWriter(log_dir="runs/saliency_gnn")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    writer.close()


# Testing loop
def test_gnn(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in data_loader:
            out = model(data.x, data.edge_index, data.edge_attr).squeeze()
            loss = criterion(out, data.y)
            total_loss += loss.item()
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Compute performance metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Test Loss: {avg_loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

    return avg_loss, mse, r2


# Data preparation
mesh_dir = "SchellingData/SchellingData/Meshes/"
ground_truth_dir = "SchellingData/SchellingData/Distributions/"
mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.off')]

dataset = []
for mesh_file in mesh_files:
    mesh_path = os.path.join(mesh_dir, mesh_file)
    ground_truth_path = os.path.join(ground_truth_dir, mesh_file.replace('.off', '.val'))
    if not os.path.exists(ground_truth_path):
        continue
    mesh = trimesh.load(mesh_path)
    vertices, faces = mesh.vertices, mesh.faces
    graph = build_mesh_graph(vertices, faces)
    normals = estimate_normals_pca_graph(vertices, graph)
    curvature = estimate_curvature_multi_scale(vertices, normals, graph)
    dk, dn = compute_dk_dn_graph(graph, vertices, curvature, normals)
    data = create_graph_for_gnn(vertices, faces, dk, dn)
    ground_truth = np.loadtxt(ground_truth_path)
    data.y = torch.tensor(ground_truth, dtype=torch.float)
    dataset.append(data)

# Split dataset and train
train_dataset = dataset[:int(0.8 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

input_dim = train_dataset[0].x.size(1)
hidden_dim = 64
output_dim = 1
model = SaliencyGNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the model
train_gnn(model, train_loader, optimizer, criterion)

# Test the model
test_gnn(model, test_loader, criterion)
