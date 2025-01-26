import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pyvista as pv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to compute non-parametric curvature (PCA-based curvature)
def compute_nonparam_curvature(mesh):
    # Use PCA to estimate curvature at each vertex
    curvature = np.zeros(mesh.vertices.shape[0])
    kdtree = cKDTree(mesh.vertices)
    
    for i, vertex in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(vertex, r=0.02)  # Radius to consider neighboring vertices
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
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    kdtree = cKDTree(mesh.vertices)
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=0.02)  # Larger radius for more neighborhood data
        if len(neighbors) < 1:
            continue
        
        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.mean(curvature_diff)
        
        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.mean(normal_diff)
        
    return dk_q, dn_q

# Function to compute saliency based on differences in curvature and normals
def compute_saliency(dk, dn):
    # Adjust saliency formula to enhance discriminative power
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    return S_q

# Function to generate descriptors for matching
def  generate_descriptors(mesh):
    vertex_normals = mesh.vertex_normals
    curvature = compute_nonparam_curvature(mesh)
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
    saliency = compute_saliency(dk, dn)
    
    descriptors = np.vstack((curvature, dn, saliency)).T
    
    return descriptors

# Matching function that compares two meshes based on saliency descriptors using cosine similarity
def match_meshes(mesh_query, meshes):
    # Generate descriptors for the query mesh
    query_descriptors = generate_descriptors(mesh_query)
    
    matching_scores = []
    file_count = 0

    for mesh in meshes:
        descriptors = generate_descriptors(mesh)
        
        similarity = cosine_similarity(query_descriptors, descriptors)
        
        match_score = np.mean(np.max(similarity, axis=1))  # Mean of max similarity per vertex
        file_count += 1

        logging.info(f"meshes processed: {file_count} / {len(meshes)}")

        matching_scores.append((mesh, match_score))

    # Sort meshes by their match scores in descending order (best matches first)
    matching_scores.sort(key=lambda x: x[1], reverse=True)

    return matching_scores

# Function to evaluate retrieval accuracy (precision and recall)
def evaluate_accuracy(retrieved_meshes, actual_class, k):
    """Evaluate precision and recall on top-k retrieved meshes."""
    top_k_retrieved = retrieved_meshes[:k]
    
    correct_count = sum(1 for mesh, _ in top_k_retrieved if mesh.metadata['class_id'] == actual_class)
    precision = correct_count / k if k > 0 else 0
    total_in_class = sum(1 for mesh, _ in retrieved_meshes if mesh.metadata['class_id'] == actual_class)
    recall = correct_count / total_in_class if total_in_class > 0 else 0
    
    return precision, recall

# Function to get matched and unmatched names
def get_matched_unmatched_meshes(retrieved_meshes, all_meshes_with_label, k):
    top_k_retrieved_names = [os.path.basename(mesh.metadata['name']) for mesh, _ in retrieved_meshes[:k]]
    
    # Get the set of all mesh names with the same label
    all_names_with_label = [os.path.basename(mesh.metadata['name']) for mesh in all_meshes_with_label]
    
    # Unmatched are those in the set difference of all meshes with this label and top-k matches
    unmatched_mesh_names = list(set(all_names_with_label) - set(top_k_retrieved_names))
    
    return top_k_retrieved_names, unmatched_mesh_names

# Function to process and match a query mesh
def process_and_match(query_mesh_path, dataset_mesh_dir, k=20):
    # Load the query mesh
    query_mesh = trimesh.load(query_mesh_path)
    query_mesh_index = int(os.path.basename(query_mesh_path).split('.')[0])
    query_mesh.metadata = {'class_id': get_class_id(query_mesh_index), 'name': os.path.basename(query_mesh_path)}

    # Load all meshes in the dataset directory
    dataset_meshes = []
    for filename in os.listdir(dataset_mesh_dir):
        if filename.endswith('.off'):
            mesh_path = os.path.join(dataset_mesh_dir, filename)
            mesh = trimesh.load(mesh_path)
            class_id = get_class_id(int(filename.split('.')[0]))  # Extracting class ID based on filename
            mesh.metadata = {'class_id': class_id, 'name': filename}
            dataset_meshes.append(mesh)
            
            logging.info(f"Loaded mesh: {filename} with class ID: {class_id}")

    logging.info(f"Total meshes loaded: {len(dataset_meshes)}")

    # Match the query mesh against all meshes
    matching_scores = match_meshes(query_mesh, dataset_meshes)

    # Get the top-k retrieved meshes (sorted by similarity score)
    retrieved_meshes = matching_scores[:k]

    # Find actual class of the query mesh
    actual_class = query_mesh.metadata['class_id']

    # Filter all dataset meshes that belong to the same class
    all_meshes_with_same_class = [mesh for mesh in dataset_meshes if mesh.metadata['class_id'] == actual_class]

    # Get matched and unmatched mesh names
    matched_names, unmatched_names = get_matched_unmatched_meshes(retrieved_meshes, all_meshes_with_same_class, k)

    logging.info(f"Top-{k} matched meshes: {matched_names}")
    logging.info(f"Unmatched meshes: {unmatched_names}")

    # Evaluate precision and recall
    precision, recall = evaluate_accuracy(retrieved_meshes, actual_class, k)
    
    logging.info(f"Precision@{k}: {precision}")
    logging.info(f"Recall@{k}: {recall}")

def get_class_id(mesh_index):
    """Map mesh index to class ID."""
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
        return 10 # Hand
    elif 201 <= mesh_index <= 220:
        return 11 # Plier
    elif 221 <= mesh_index <= 240:
        return 12 # Fish
    elif 241 <= mesh_index <= 260:
        return 13 # Bird
    elif 281 <= mesh_index <= 300:
        return 14 # Armadillo
    elif 301 <= mesh_index <= 320:
        return 15 # Bust
    elif 321 <= mesh_index <= 340:
        return 16 # Mech
    elif 341 <= mesh_index <= 360:
        return 17 # Bearing
    elif 361 <= mesh_index <= 380:
        return 18 # Vase
    elif 381 <= mesh_index <= 400:
        return 19 # Fourleg
    return None

# Directory paths
dataset_mesh_dir = 'mesheg/models/MeshsegBenchmark-1.0/data/off/'

# Example query
query_mesh_path = "mesheg/models/MeshsegBenchmark-1.0/data/off/1.off"

# Process the query and match
process_and_match(query_mesh_path, dataset_mesh_dir)
