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
def estimate_normals_pca(vertices, graph, min_neighbors=3, scales=[0.001, 0.005, 0.01]):
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
        
        if len(neighbors) < min_neighbors:
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

# Compute saliency metric
def compute_saliency(dn_q, dk_q):
    return 1 - np.exp(-dn_q) - np.exp(-dk_q)

# Build a graph for mesh vertices
def build_graph(vertices, faces, weighting="gaussian", params=None):
    graph = nx.Graph()
    for face in faces:
        for i in range(3):
            v1_idx, v2_idx = face[i], face[(i + 1) % 3]
            v1, v2 = vertices[v1_idx], vertices[v2_idx]
            weight = compute_weight(v1, v2, weighting, params)
            if weight > 0:
                graph.add_edge(v1_idx, v2_idx, weight=weight)
    return graph

# Load mesh data into PyTorch Geometric format
def load_mesh_data(mesh_dir, ground_truth_dir, max_meshes=380):
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
        graph = build_graph(vertices, faces, params={"sigma": 0.02, "radius": 0.1})

        # Estimate normals and curvature
        vertex_normals, eigenvalues_list = estimate_normals_pca(vertices, graph)
        curvatures = estimate_curvature(vertices, vertex_normals, graph)

        # Compute dn and dk metrics
        dn = np.linalg.norm(vertex_normals - np.mean(vertex_normals, axis=0), axis=1)
        dk = np.abs(curvatures - np.mean(curvatures))

        # Compute saliency
        saliency = compute_saliency(dn, dk)

        # Create PyTorch Geometric Data
        # Feature vector includes vertices, normals, curvature, and eigenvalues
        x = torch.tensor(np.hstack([vertices, saliency[:, None], vertex_normals, curvatures[:, None], eigenvalues_list]), dtype=torch.float)
        edges = list(graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor([graph[u][v]['weight'] for u, v in edges], dtype=torch.float)

        ground_truth = torch.tensor(np.loadtxt(ground_truth_path), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=ground_truth, mesh_path=mesh_path, ground_truth_path=ground_truth_path)
        dataset.append(data)

    return dataset

# Visualize a mesh
def visualize_mesh_pyvista(vertices, faces, saliency=None):
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)
    if saliency is not None:
        mesh.point_data['Saliency'] = saliency
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='Saliency', cmap='Reds')
    plotter.show() 

# Define the GNN model using GraphSAGE
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
		

# Main script
if __name__ == "__main__":
    mesh_dir = "SchellingData/SchellingData/Meshes/"
    ground_truth_dir = "SchellingData/SchellingData/Distributions/"

    dataset = load_mesh_data(mesh_dir, ground_truth_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    input_dim = 11
    hidden_dim = 64
    output_dim = 1
    model = GNN(input_dim, hidden_dim, output_dim, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(100):
        total_loss = 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            predictions = torch.sigmoid(model(batch))
            loss = criterion(predictions, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            mesh = trimesh.load(data.mesh_path[0])
            saliency = torch.sigmoid(model(data)).squeeze().numpy()
            with open(data.ground_truth_path[0], 'r') as file:
                ground_truth = np.array([float(line.strip()) for line in file])
            mean_gt = np.mean(ground_truth)
            std_gt = np.std(ground_truth)
            mean_saliency = np.mean(saliency)
            std_saliency = np.std(saliency)
            saliency_transformed = (saliency - mean_saliency) * (std_gt / std_saliency) + mean_gt
            saliency_transformed = np.clip(saliency_transformed, 0, 1)
            print(saliency_transformed)
            np.savetxt("saliency_values.txt", saliency_transformed)
            visualize_mesh_pyvista(mesh.vertices, mesh.faces, saliency=saliency_transformed)
