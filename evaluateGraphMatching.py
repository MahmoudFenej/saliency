import torch
import os
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import trimesh
from sklearn.model_selection import train_test_split
import logging
from torch_geometric.nn import global_mean_pool
from scipy.spatial import Delaunay

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to compute non-parametric curvature using PCA
def compute_nonparam_curvature(mesh):
    logging.info("Computing non-parametric curvature.")
    curvature = np.zeros(mesh.vertices.shape[0])
    kdtree = cKDTree(mesh.vertices)
    
    for i, vertex in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(vertex, r=0.02)
        if len(neighbors) < 3:
            continue
        
        neighbor_points = mesh.vertices[neighbors]
        pca = PCA(n_components=3)
        pca.fit(neighbor_points)
        
        # Curvature is estimated as the ratio of the smallest to the largest PCA eigenvalue
        curvature[i] = pca.explained_variance_[2] / (pca.explained_variance_[0] + 1e-8)  # Avoid division by zero
    
    curvature_magnitude = np.abs(curvature)
    return curvature_magnitude

# Function to compute differences in curvature and normal for each vertex
def compute_dk_dn(mesh, curvature, vertex_normals):
    logging.info("Computing differences in curvature and normals.")
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=0.02)
        if len(neighbors) < 1:
            continue
        
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.mean(curvature_diff)
        
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.mean(normal_diff)
        
    return dk_q, dn_q

# Function to compute saliency based on differences in curvature and normals
def compute_saliency(dk, dn):
    logging.info("Computing saliency based on curvature and normal differences.")
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))  # Enhance saliency formula
    return S_q

# Function to generate descriptors for matching
def generate_descriptors(mesh):
    logging.info("Generating descriptors for the mesh.")
    vertex_normals = mesh.vertex_normals
    curvature = compute_nonparam_curvature(mesh)
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
    saliency = compute_saliency(dk, dn)
    
    descriptors = np.vstack((curvature, dn, saliency)).T
    return descriptors

# Map mesh index to class ID
def get_class_id(mesh_index):
    if 1 <= mesh_index <= 20:
        return 1  # Human
    elif 21 <= mesh_index <= 40:
        return 2  # Cup
    elif 41 <= mesh_index <= 60:
        return 3  # Glasses
    elif 61 <= mesh_index <= 80:
        return 4  # Airplane
    elif 81 <= mesh_index <= 100:
        return 5  # Ant
    elif 101 <= mesh_index <= 120:
        return 6  # Chair
    elif 121 <= mesh_index <= 140:
        return 7  # Octopus
    elif 141 <= mesh_index <= 160:
        return 8  # Table
    elif 161 <= mesh_index <= 180:
        return 9  # Teddy
    elif 181 <= mesh_index <= 200:
        return 10  # Hand
    elif 201 <= mesh_index <= 220:
        return 11  # Plier
    elif 221 <= mesh_index <= 240:
        return 12  # Fish
    elif 241 <= mesh_index <= 260:
        return 13  # Bird
    elif 281 <= mesh_index <= 300:
        return 14  # Armadillo
    elif 301 <= mesh_index <= 320:
        return 15  # Bust
    elif 321 <= mesh_index <= 340:
        return 16  # Mech
    elif 341 <= mesh_index <= 360:
        return 17  # Bearing
    elif 361 <= mesh_index <= 380:
        return 18  # Vase
    elif 381 <= mesh_index <= 400:
        return 19  # Fourleg
    return None

def preprocess_mesh_with_descriptors(mesh):
    logging.info("Preprocessing mesh with descriptors.")

    descriptors = generate_descriptors(mesh)

    try:
        delaunay = Delaunay(mesh.vertices)
    except Exception as e:
        raise ValueError(f"Delaunay triangulation failed: {e}")

    edges = set()
    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add(tuple(sorted((simplex[i], simplex[j]))))  # Avoid duplicate edges

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    if edge_index.size(0) != 2:
        raise ValueError(f"Invalid edge_index dimensions: {edge_index.shape}")

    # Convert descriptors to PyTorch tensor
    features = torch.tensor(descriptors, dtype=torch.float)

    return Data(x=features, edge_index=edge_index)


def build_dataset_with_descriptors(dataset_dir, r=0.02):
    data_list = []
    labels = []
    logging.info("Building dataset from meshes in directory: %s", dataset_dir)
    file_count = 0
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.off'):
            index = int(filename.split('.')[0])
            class_id = get_class_id(index)
            if class_id is None:
                continue
            mesh = trimesh.load(os.path.join(dataset_dir, filename))
            data = preprocess_mesh_with_descriptors(mesh)
            data.class_id = class_id
            data_list.append(data)
            labels.append(class_id)
            file_count += 1
                # if file_count >= 100:
                #     break

    return data_list, labels

class MeshEmbeddingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MeshEmbeddingGNN, self).__init__()
        self.conv1 = torch_geometric.nn.GraphConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GraphConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class MatchingModel(torch.nn.Module):
    def __init__(self, gnn, embedding_dim):
        super(MatchingModel, self).__init__()
        self.gnn = gnn
        self.embedding_dim = embedding_dim

    def forward(self, data1, data2):
        emb1 = self.gnn(data1)
        emb2 = self.gnn(data2)
        
        emb1 = global_mean_pool(emb1, data1.batch)
        emb2 = global_mean_pool(emb2, data2.batch)

        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity

def train(model, optimizer, train_loader, criterion):
    model.train()
    for data in train_loader:
        data1, data2, label = data
        optimizer.zero_grad()
        similarity = model(data1, data2)
        loss = criterion(similarity, label.float())
        loss.backward()
        optimizer.step()

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    logging.info("Testing model.")
    with torch.no_grad():
        # Only test on the first pair
        data1, data2, label = test_loader[0]  # Get the first pair
        similarity = model(data1, data2)
        predicted = (similarity > 0.5).float()
        correct += (predicted == label).sum().item()
        total += 1  # Increment by 1 for each label
    
    accuracy = correct / total if total > 0 else 0
    logging.info(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def generate_pairs(data, labels):
    pairs = []
    logging.info("Generating pairs of meshes for matching.")
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            pair1 = data[i]
            pair2 = data[j]
            label = 1 if labels[i] == labels[j] else 0
            pairs.append((pair1, pair2, label))
    return pairs

def main():
    dataset_dir = 'mesheg/models/MeshsegBenchmark-1.0/data/off'
    
    logging.info("Starting dataset processing.")
    # Use descriptor-based preprocessing
    data_list, labels = build_dataset_with_descriptors(dataset_dir, r=0.02)

    # Split data
    logging.info("Splitting dataset into training and test sets.")
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_list, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Generate positive and negative pairs
    train_pairs = generate_pairs(train_data, train_labels)
    test_pairs = generate_pairs(test_data, test_labels)

    # Create data loaders for pairs
    train_loader = DataLoader(train_pairs, batch_size=2, shuffle=True)
    test_loader = test_pairs

    # Initialize model
    logging.info("Initializing model.")
    gnn = MeshEmbeddingGNN(in_channels=3, hidden_channels=64, out_channels=128)
    model = MatchingModel(gnn, embedding_dim=128)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train the model
    logging.info("Training phase started.")
    for epoch in range(10):
        train(model, optimizer, train_loader, criterion)
        logging.info(f'Epoch {epoch+1}: Training complete.')
        
        # Test the model
        accuracy = test(model, test_loader)
        logging.info(f'Epoch {epoch+1}: Test accuracy = {accuracy:.4f}')

if __name__ == '__main__':
    main()
