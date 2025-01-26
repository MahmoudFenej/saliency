import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from karateclub import Graph2Vec
from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull

# Load the 3D mesh from an OFF file
mesh = trimesh.load("person.off")
vertices = mesh.vertices
faces = mesh.faces

# Compute face and vertex normals
face_normals = mesh.face_normals
vertex_normals = mesh.vertex_normals

# Function to compute non-parametric curvature using Laplacian smoothing
def compute_nonparam_curvature(mesh):
    L = trimesh.smoothing.laplacian_calculation(mesh)
    curvature = L.dot(mesh.vertices)
    curvature_magnitude = np.linalg.norm(curvature, axis=1)
    return curvature_magnitude

# Function to compute differences in curvature and normal for each vertex
def compute_dk_dn(mesh, curvature, vertex_normals):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=0.1)
        if len(neighbors) < 1:
            continue
        
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.mean(curvature_diff)
        
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.mean(normal_diff)
        
    return dk_q, dn_q

# Function to compute saliency based on differences in curvature and normals
def compute_saliency(dk, dn):
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    return S_q

# Compute curvature, differences, and saliency
curvature = compute_nonparam_curvature(mesh)
dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
saliency = compute_saliency(dk, dn)

# Define a saliency threshold
saliency_threshold = 0.7

# Filter vertices based on the saliency threshold
filtered_indices = np.where(saliency > saliency_threshold)[0]
filtered_vertices = vertices[filtered_indices]

# Create a mapping from the original vertex indices to the filtered indices
index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(filtered_indices)}

# Construct a graph from the filtered vertices and edges
def construct_graph_from_salient_parts(vertices, faces, filtered_indices, index_mapping):
    G = nx.Graph()
    
    # Add nodes for filtered vertices
    for idx in filtered_indices:
        G.add_node(index_mapping[idx], pos=vertices[idx], saliency=saliency[idx])
    
    # Add edges for faces that connect only the filtered vertices
    for face in faces:
        # Check if all vertices of the face are in the filtered list
        if all(v in filtered_indices for v in face):
            for i in range(3):
                G.add_edge(index_mapping[face[i]], index_mapping[face[(i + 1) % 3]])
    
    return G

# Extract graph-based features
def extract_graph_features(G):
    # Node features
    degrees = np.array([G.degree(n) for n in G.nodes])
    centralities = np.array(list(nx.degree_centrality(G).values()))
    saliency_values = np.array(list(nx.get_node_attributes(G, 'saliency').values()))
    
    # Graph embedding
    embedding_model = Graph2Vec(dimensions=128)
    embedding_model.fit([G])
    graph_embedding = embedding_model.get_embedding()
    
    # Aggregate node features (mean, std, skewness, kurtosis)
    features = {
        "degree_mean": np.mean(degrees),
        "degree_std": np.std(degrees),
        "degree_skew": skew(degrees),
        "degree_kurtosis": kurtosis(degrees),
        "centrality_mean": np.mean(centralities),
        "centrality_std": np.std(centralities),
        "centrality_skew": skew(centralities),
        "centrality_kurtosis": kurtosis(centralities),
        "saliency_mean": np.mean(saliency_values),
        "saliency_std": np.std(saliency_values),
        "saliency_skew": skew(saliency_values),
        "saliency_kurtosis": kurtosis(saliency_values),
        "graph_embedding": graph_embedding
    }
    
    return features

# Extract curvature and saliency statistics
def extract_geometric_features(curvature, saliency, filtered_indices):
    filtered_curvature = curvature[filtered_indices]
    filtered_saliency = saliency[filtered_indices]
    
    features = {
        "curvature_mean": np.mean(filtered_curvature),
        "curvature_std": np.std(filtered_curvature),
        "curvature_skew": skew(filtered_curvature),
        "curvature_kurtosis": kurtosis(filtered_curvature),
        "saliency_mean": np.mean(filtered_saliency),
        "saliency_std": np.std(filtered_saliency),
        "saliency_skew": skew(filtered_saliency),
        "saliency_kurtosis": kurtosis(filtered_saliency)
    }
    
    return features

# Combine all features into a single vector
def combine_features(graph_features, geometric_features):
    feature_vector = [
        graph_features["degree_mean"],
        graph_features["degree_std"],
        graph_features["degree_skew"],
        graph_features["degree_kurtosis"],
        graph_features["centrality_mean"],
        graph_features["centrality_std"],
        graph_features["centrality_skew"],
        graph_features["centrality_kurtosis"],
        graph_features["saliency_mean"],
        graph_features["saliency_std"],
        graph_features["saliency_skew"],
        graph_features["saliency_kurtosis"],
        geometric_features["curvature_mean"],
        geometric_features["curvature_std"],
        geometric_features["curvature_skew"],
        geometric_features["curvature_kurtosis"],
        geometric_features["saliency_mean"],
        geometric_features["saliency_std"],
        geometric_features["saliency_skew"],
        geometric_features["saliency_kurtosis"],
    ]
    
    # Append graph embedding
    feature_vector.extend(graph_features["graph_embedding"])
    
    return np.array(feature_vector)

# Extract spatial relationship features
def extract_spatial_features(vertices, filtered_indices):
    filtered_vertices = vertices[filtered_indices]

    # Pairwise distances
    pairwise_distances = distance_matrix(filtered_vertices, filtered_vertices)
    pairwise_distances = pairwise_distances[np.triu_indices(len(filtered_vertices), k=1)]
    
    # Bounding box
    bbox_min = np.min(filtered_vertices, axis=0)
    bbox_max = np.max(filtered_vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_volume = np.prod(bbox_size)
    
    # Centroid and distribution
    centroid = np.mean(filtered_vertices, axis=0)
    distances_to_centroid = np.linalg.norm(filtered_vertices - centroid, axis=1)
    
    # Convex hull
    hull = ConvexHull(filtered_vertices)
    hull_area = hull.area
    hull_volume = hull.volume
    
    # Aggregate spatial features
    spatial_features = {
        "pairwise_distances_mean": np.mean(pairwise_distances),
        "pairwise_distances_std": np.std(pairwise_distances),
        "pairwise_distances_skew": skew(pairwise_distances),
        "pairwise_distances_kurtosis": kurtosis(pairwise_distances),
        "bbox_size_x": bbox_size[0],
        "bbox_size_y": bbox_size[1],
        "bbox_size_z": bbox_size[2],
        "bbox_volume": bbox_volume,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2],
        "distances_to_centroid_mean": np.mean(distances_to_centroid),
        "distances_to_centroid_std": np.std(distances_to_centroid),
        "hull_area": hull_area,
        "hull_volume": hull_volume,
    }
    
    return spatial_features

# Construct and extract features
G = construct_graph_from_salient_parts(vertices, faces, filtered_indices, index_mapping)
graph_features = extract_graph_features(G)
geometric_features = extract_geometric_features(curvature, saliency, filtered_indices)
feature_vector = combine_features(graph_features, geometric_features)

# Print the feature vector
print("Feature Vector:")
print(len(feature_vector))
