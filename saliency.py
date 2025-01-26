import numpy as np
import pyvista as pv
from scipy import stats
from scipy.spatial import KDTree

class CurvatureKernel:
    def __init__(self, alpha, roughness, min_points=4):
        self._alpha = alpha
        self._roughness = roughness
        self._minPoints_neighbourhood = min_points

    def process(self, pt, neighbours):
        epsilon = stats.norm.ppf(1 - self._alpha / 2) * self._roughness

        if len(neighbours) < self._minPoints_neighbourhood:
            print('Point does not have enough neighbours, curvature set to zero')
            return 0

        normal_pt = pt['normal']
        P_q = np.eye(3) - np.outer(normal_pt, normal_pt)
        pt_projected = P_q.dot(pt['coords'])

        n_xyz = np.array([n['coords'] for n in neighbours])
        proj_np = normal_pt.dot((pt['coords'].reshape(-1, 1) - n_xyz.T))  # Reshape for correct broadcasting
        proj_tangent = P_q.dot(n_xyz.T).T

        cg = np.mean(proj_tangent - pt_projected, axis=0)
        if np.linalg.norm(cg) > 1:
            return 0

        sum_proj = np.sum(proj_np) / (len(neighbours) - 1)

        if np.abs(sum_proj) < epsilon:
            return 0
        else:
            return sum_proj


class SaliencyKernel:
    def __init__(self, rho_neighbourhood, sigma_neighbourhood, sigma_normal=0.01, sigma_curvature=0.1,
                 alpha=0.05, min_points=4):
        self._alpha = alpha
        self._sigma_curvature = sigma_curvature
        self._sigma_normal = sigma_normal
        self._sigma_neighbourhood = sigma_neighbourhood
        self._rho_neighbourhood = rho_neighbourhood
        self._minPoints_neighbourhood = min_points

    def process(self, pt, neighbours):
        if len(neighbours) < self._minPoints_neighbourhood:
            print('Point does not have enough neighbours, dk and dn set to zero')
            return 0, 0

        pt_normal = pt['normal']
        pt_curvature = pt['curvature']

        n_xyz = np.array([n['coords'] for n in neighbours])
        n_curvature = np.array([n['curvature'] for n in neighbours])
        n_normal = np.array([n['normal'] for n in neighbours])

        dxyz = pt['coords'] - n_xyz
        dist = np.sum(dxyz**2, axis=1)
        dn = 1 - np.einsum('ij,j->i', n_normal, pt_normal)  # Use einsum for dot product along axis
        dk = pt_curvature - n_curvature

        rho2 = self._rho_neighbourhood**2
        neighbourhood_weights = np.zeros(dist.shape[0])

        neighbourhood_weights[dist < rho2] = 1 / rho2 * dist[dist < rho2]
        neighbourhood_weights[dist > rho2] = -1 / rho2 * dist[dist > rho2] + 2
        neighbourhood_weights[neighbourhood_weights < 0] = 0

        active_w = neighbourhood_weights > 0.01
        number_active = int(np.sum(active_w))
        chi2_table = stats.chi2.ppf(self._alpha, number_active)

        chi2_normal = (len(neighbours) - 1) * dn.std() / self._sigma_normal
        chi2_curvature = (len(neighbours) - 1) * dk.std() / self._sigma_curvature

        num_zeros_normals = np.sum(np.abs(dn[active_w]) < 0.016)
        num_zeros_curvature = np.sum(np.abs(dk[active_w]) <= 0.04)

        if chi2_normal < chi2_table or num_zeros_normals > 0.6 * number_active:
            dn = 0
        else:
            dn = np.sum(np.abs(dn) * neighbourhood_weights) / np.sum(neighbourhood_weights)

        if chi2_curvature < chi2_table or num_zeros_curvature > 0.6 * number_active:
            dk = 0
        else:
            dk = np.abs(np.sum(dk * neighbourhood_weights) / np.sum(neighbourhood_weights))

        return dk, dn


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF file')

        n_verts, n_faces, _ = tuple(map(int, file.readline().strip().split(' ')))

        vertices = [list(map(float, file.readline().strip().split(' '))) for _ in range(n_verts)]
        faces = [list(map(int, file.readline().strip().split(' ')[1:])) for _ in range(n_faces)]

    return np.array(vertices), faces


def compute_normals_and_curvatures(vertices, faces):
    # Placeholder function for normal and curvature computation
    # Replace this with actual computation as needed
    normals = np.array([[0, 0, 1]] * len(vertices))
    curvatures = np.random.rand(len(vertices))  # Random curvature for demonstration
    return normals, curvatures

def compute_saliency(vertices, faces):
    # Compute normals and curvatures
    _, curvatures = compute_normals_and_curvatures(vertices, faces)

    # Normalize curvature values to fit in the [0, 1] range
    curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))

    return curvatures


def visualize_mesh_pyvista(vertices, faces, saliency=None):
    # Create a PyVista mesh object from vertices and faces
    faces_flat = np.hstack([np.hstack([len(face), *face]) for face in faces])
    mesh = pv.PolyData(vertices, faces_flat)

    if saliency is not None:
        # Add saliency as a scalar array to the mesh for coloring
        mesh.point_data['Saliency'] = saliency

    # Create a plotter object for interactive 3D visualization
    plotter = pv.Plotter(window_size=[800, 600])  # Adjust size as needed

    # Define a custom color map where higher saliency values are mapped to red
    cmap = 'Reds' if saliency is not None else 'viridis'

    # Add the mesh to the plotter with the saliency scalars
    if saliency is not None:
        plotter.add_mesh(mesh, scalars='Saliency', cmap=cmap)
    else:
        plotter.add_mesh(mesh, color='viridis')

    # Display the interactive window
    plotter.show()


def main():
    vertices, faces = load_off_file('SchellingData/SchellingData/Meshes/1.off')
    saliency = compute_saliency(vertices, faces)
    saliency = np.where(saliency < 0.9, 0, saliency)

    print(saliency)
    visualize_mesh_pyvista(vertices, faces, saliency)

if __name__ == "__main__":
    main()
