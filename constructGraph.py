import numpy as np 
import trimesh
import pyvista as pv
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.metrics import precision_score, recall_score

# Load the 3D mesh
mesh = trimesh.load("SchellingData/SchellingData/Meshes/2.off")
ground_truth_path = "SchellingData/SchellingData/Distributions/2.val"

with open(ground_truth_path, 'r') as file:
    ground_truth = np.array([float(line.strip()) for line in file])

vertices = mesh.vertices
faces = mesh.faces

# Function to compute neighbors using Delaunay triangulation
def compute_neighbors_delaunay(vertices):
    delaunay = Delaunay(vertices[:, :2])  # Using only the x, y coordinates for 2D triangulation
    neighbors = {i: set() for i in range(len(vertices))}
    
    for simplex in delaunay.simplices:
        for i in simplex:
            neighbors[i].update(simplex)
            neighbors[i].remove(i)  # Remove self from neighbors
    
    # Convert sets back to lists
    for key in neighbors:
        neighbors[key] = list(neighbors[key])
    
    return neighbors

# Compute normals using covariance-based PCA
def compute_normals_pca(vertices, neighbors_dict):
    normals = np.zeros(vertices.shape)
    for i, neighbors in neighbors_dict.items():
        if len(neighbors) < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(vertices[neighbors] - vertices[i])
        normals[i] = pca.components_[-1]
    return normals

# Patch Length
def compute_patch_length(vertices, neighbors_dict):
    patch_lengths = np.zeros(len(vertices))
    for i, neighbors in neighbors_dict.items():
        if len(neighbors) > 0:
            patch_lengths[i] = max(np.linalg.norm(vertices[j] - vertices[i]) ** 2 for j in neighbors)
    return patch_lengths

# Compute curvature
def compute_curvature(vertices, faces):
    curvature = np.zeros(len(vertices))
    normals = np.zeros_like(vertices)
    areas = np.zeros(len(vertices))
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n)  # Normalize the normal
        area = np.linalg.norm(n) / 2  # Area of triangle
        
        for idx in face:
            normals[idx] += n
            areas[idx] += area
    
    for i in range(len(normals)):
        if areas[i] > 0:
            normals[i] /= np.linalg.norm(normals[i])
    
    for i in range(len(vertices)):
        if areas[i] > 0:
            curvature[i] = np.linalg.norm(normals[i]) / areas[i]
    
    return curvature

# Compute scale parameter
def compute_scale_parameter(vertices, neighbors_dict):
    scale_parameters = np.zeros(len(vertices))
    for i, neighbors in neighbors_dict.items():
        if len(neighbors) > 0:
            scale_parameters[i] = max(np.linalg.norm(vertices[j] - vertices[i]) for j in neighbors)
    return scale_parameters

# Compute K based on normalized curvature
def compute_K(curvature):
    return (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-5)

# Compute DF (distance function) and normalize
def compute_DF(vertices, neighbors_dict):
    df = np.zeros((len(vertices), len(vertices)))
    for i, neighbors in neighbors_dict.items():
        for j in neighbors:
            df[i, j] = np.linalg.norm(vertices[i] - vertices[j])
    # Normalize DF
    max_distance = np.max(df)
    if max_distance > 0:
        df /= max_distance
    return df

# Compute He based on normalized normals
def compute_He(normals):
    he = np.linalg.norm(normals, axis=1)
    return (he - np.min(he)) / (np.max(he) - np.min(he) + 1e-5)

# Similarity function
def similarity_function(v_i, v_j, K, DF, He, sigma):
    diff_he = He[v_j] - He[v_i]
    distance = DF[v_j, v_i] if DF[v_j, v_i] > 0 else 1e-5  # Prevent division by zero
    return np.exp((K[v_j] * (diff_he ** 2)) / (sigma[v_i] ** 2 + distance))

# Compute saliency
def compute_saliency(vertices, neighbors_dict, K, DF, He, sigma):
    saliency = np.zeros(len(vertices))
    for i in range(len(vertices)):
        total_similarity = 0
        for j in neighbors_dict[i]:
            total_similarity += similarity_function(i, j, K, DF, He, sigma)
        saliency[i] = total_similarity / len(neighbors_dict[i]) if neighbors_dict[i] else 0
    
    # Normalize saliency values
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-5)
    return saliency

# Visualization function
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

# Compute neighbors using Delaunay triangulation
neighbors = compute_neighbors_delaunay(vertices)

# Compute normals
normals = compute_normals_pca(vertices, neighbors)

# Compute curvature
curvature = compute_curvature(vertices, faces)

# Compute patch length
patch_lengths = compute_patch_length(vertices, neighbors)

# Compute K based on curvature
K = compute_K(curvature)

# Compute DF (distance function)
DF = compute_DF(vertices, neighbors)

# Compute He based on normals
He = compute_He(normals)

# Compute scale parameters (σ)
sigma = compute_scale_parameter(vertices, neighbors)

# Compute saliency
saliency = compute_saliency(vertices, neighbors, K, DF, He, sigma)
saliency = np.array(saliency > 0.9)

# Visualize results
visualize_mesh_pyvista(vertices, faces, saliency)
visualize_mesh_pyvista(vertices, faces, ground_truth)

# Binarize saliency for precision and recall
def binarize_saliency(saliency_array, threshold=0.5):
    binary_mask = np.where(saliency_array > threshold, 1, 0)
    return binary_mask

predicted_mask = binarize_saliency(saliency, threshold=0.5)
ground_truth_mask = binarize_saliency(ground_truth, threshold=0)

# Flatten the masks for precision and recall
predicted_mask_flat = predicted_mask.flatten()
ground_truth_mask_flat = ground_truth_mask.flatten()

# Compute Precision and Recall
precision = precision_score(ground_truth_mask_flat, predicted_mask_flat)
recall = recall_score(ground_truth_mask_flat, predicted_mask_flat)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
