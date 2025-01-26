import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import vedo
from vedo import show, Points

# Load the mesh
MODELNET_PATH = "ModelNet10"
categories = [folder for folder in os.listdir(MODELNET_PATH) if os.path.isdir(os.path.join(MODELNET_PATH, folder))]

def load_modelnet_files(dataset_type='train', max_files=100):
    meshes = []
    labels = []
    file_count = 0
    
    for category in categories:
        category_dir = os.path.join(MODELNET_PATH, category, dataset_type)
        for filename in os.listdir(category_dir):
            if filename.endswith('.off'):
                mesh = trimesh.load(os.path.join(category_dir, filename))
                meshes.append(mesh)
                labels.append(category)
                file_count += 1
            if file_count >= max_files:
                break
                
    return meshes, labels

def compute_curvatures(mesh):
    # Assuming curvatures are provided by trimesh (using placeholder here)
    k1 = np.random.random(len(mesh.vertices))  # Placeholder for first curvature
    k2 = np.random.random(len(mesh.vertices))  # Placeholder for second curvature
    
    SJ = (2 / np.pi) * np.arctan((k2 + k1) / (k2 - k1))
    CJ = np.sqrt((k1**2 + k2**2) / 2)
    G = k1 * k2
    M = 0.5 * (k1 + k2)
    
    return SJ, CJ, G, M

def compute_dynamic_radius(mesh, vertices, scaling_factor=0.1):
    kdtree = cKDTree(vertices)
    dists, _ = kdtree.query(vertices, k=2)
    avg_dist = np.mean(dists[:, 1])
    bounding_box = mesh.bounding_box_oriented.extents
    bbox_diag = np.linalg.norm(bounding_box)
    r = scaling_factor * bbox_diag if bbox_diag > 0 else avg_dist * scaling_factor
    return r

def compute_dk_dn(mesh, curvature, vertex_normals):
    dn_q = np.zeros(mesh.vertices.shape[0])
    dk_q = np.zeros(mesh.vertices.shape[0])
    
    dynamic_r = compute_dynamic_radius(mesh, mesh.vertices, scaling_factor=0.1)
    kdtree = cKDTree(mesh.vertices)
    
    for i, q in enumerate(mesh.vertices):
        neighbors = kdtree.query_ball_point(q, r=dynamic_r)
        if len(neighbors) < 1:
            continue

        curvature_diff = np.abs(curvature[i] - curvature[neighbors])
        dk_q[i] = np.mean(curvature_diff)

        normal_diff = np.linalg.norm(vertex_normals[i] - vertex_normals[neighbors], axis=1)
        dn_q[i] = np.mean(normal_diff)

    return dk_q, dn_q

def compute_saliency(dk, dn):
    S_q = 2 - (np.exp(-dn) + np.exp(-dk))
    return S_q


def extract_features_for_interest_points(mesh, saliency_threshold):
    vertex_normals = mesh.vertex_normals
    vertices = mesh.vertices
    
    # Compute saliency
    curvature = np.random.random(len(mesh.vertices))  # Replace with actual curvature computation
    dk, dn = compute_dk_dn(mesh, curvature, vertex_normals)
    saliency = compute_saliency(dk, dn)
    
    # Identify interest points
    high_saliency_indices = np.where(saliency > np.quantile(saliency, saliency_threshold))[0]
    interest_points = vertices[high_saliency_indices]
    
    # Compute curvature features for interest points
    SJ, CJ, G, M = compute_curvatures(mesh)
    
    # Filter features for interest points
    features = np.column_stack([SJ[high_saliency_indices], CJ[high_saliency_indices], G[high_saliency_indices], M[high_saliency_indices]])
    
    return features.mean(axis=0)  # Return the mean feature vector for simplicity

def process_meshes(meshes, saliency_threshold=0.4):
    feature_list = []
    for idx, mesh in enumerate(meshes):
        features = extract_features_for_interest_points(mesh, saliency_threshold)
        feature_list.append(features)
        print(f"Feature extraction done for mesh {idx + 1}/{len(meshes)}")
    return np.array(feature_list)

# Load train and test data
train_meshes, train_labels = load_modelnet_files('train')
test_meshes, test_labels = load_modelnet_files('test')

# Extract features
train_features = process_meshes(train_meshes)
test_features = process_meshes(test_meshes)

# Label encoding
label_to_int = {label: idx for idx, label in enumerate(categories)}
train_labels_encoded = np.array([label_to_int[label] for label in train_labels])
test_labels_encoded = np.array([label_to_int[label] for label in test_labels])

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

# Train and evaluate different classifiers
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    return accuracy

# SVM Classifier
svm_classifier = SVC(kernel='linear')
svm_accuracy = evaluate_classifier(svm_classifier, X_train_scaled, train_labels_encoded, X_test_scaled, test_labels_encoded)
print(f"SVM Classification accuracy: {svm_accuracy * 100:.2f}%")

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_accuracy = evaluate_classifier(rf_classifier, X_train_scaled, train_labels_encoded, X_test_scaled, test_labels_encoded)
print(f"Random Forest Classification accuracy: {rf_accuracy * 100:.2f}%")

# Neural Network Classifier
nn_classifier = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500)
nn_accuracy = evaluate_classifier(nn_classifier, X_train_scaled, train_labels_encoded, X_test_scaled, test_labels_encoded)
print(f"Neural Network Classification accuracy: {nn_accuracy * 100:.2f}%")
