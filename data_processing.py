import open3d as o3d
import numpy as np
import os


# Load and normalize mesh
def load_and_normalize_mesh(path, target_scale=100.0):
    mesh = o3d.io.read_triangle_mesh(path)

    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError("Invalid or empty mesh")

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)

    scale = target_scale / np.max(bbox.get_extent())
    mesh.scale(scale, center=(0, 0, 0))

    return mesh


# Sample point cloud
def sample_point_cloud(mesh, n_points=3000):
    pcd = mesh.sample_points_poisson_disk(n_points)
    pcd.estimate_normals()
    return pcd


# Shape diameter function using raycasting
def compute_sdf(mesh, pcd):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    sdf = np.zeros(len(points))

    for i in range(len(points)):
        p = points[i]
        n = normals[i]

        rays = np.array([
            np.hstack((p, n)),
            np.hstack((p, -n))
        ], dtype=np.float32)

        rays = o3d.core.Tensor(rays)
        ans = scene.cast_rays(rays)

        t_hits = ans['t_hit'].numpy()
        valid = t_hits[np.isfinite(t_hits)]

        sdf[i] = np.min(valid) if len(valid) > 0 else 0

    return sdf

# Surface variation https://openaccess.thecvf.com/content_cvpr_2016/papers/Hackel_Contour_Detection_in_CVPR_2016_paper.pdf
def compute_curvature(pcd, k=10):
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    curvatures = np.zeros(len(points))

    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]

        # compute covariance
        mean = neighbors.mean(axis=0)
        cov = np.dot((neighbors - mean).T, (neighbors - mean)) / k

        # eigen decomposition
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)

        # curvature estimate
        curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-8)

    return curvatures

# Normal alignment
def compute_normal_alignment(pcd, k=6):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    tree = o3d.geometry.KDTreeFlann(pcd)
    alignment = np.zeros(len(points))

    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)

        n_i = normals[i]
        neigh_normals = normals[idx]

        dots = np.abs(neigh_normals @ n_i)
        alignment[i] = np.mean(dots)

    return alignment


# Local density
def compute_density(mesh, pcd, radius_ratio=0.01):
    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    radius = radius_ratio * diag

    density = np.zeros(len(points))

    for i in range(len(points)):
        _, idx, _ = tree.search_radius_vector_3d(points[i], radius)
        density[i] = len(idx)

    return density


# Partial random
def random_dropout(full, keep_ratio=0.3):
    idx = np.random.choice(len(full), int(len(full)*keep_ratio), replace=False)
    return full[idx]

# Partial sphere cut
def cut_by_sphere(full, bbox, alpha=0.4):
    points = full[:, 0:3]

    # random center inside bbox
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = np.random.uniform(min_bound, max_bound)

    # radius from bbox diagonal
    extent = bbox.get_extent()
    diag = np.linalg.norm(extent)
    radius = alpha * diag

    # remove points inside sphere
    dist = np.linalg.norm(points - center, axis=1)
    mask = dist > radius    
    
    return full[mask]

def resample_points(data, n_points=1000):
    idx = np.random.choice(len(data), n_points, replace=len(data) < n_points)
    return data[idx]


# Full pipeline
def process_mesh(path):
    mesh = load_and_normalize_mesh(path)
    bbox = mesh.get_axis_aligned_bounding_box()

    pcd = sample_point_cloud(mesh)

    #sdf = compute_sdf(mesh, pcd)
    #normals = np.asarray(pcd.normals)
    surf_variation = compute_curvature(pcd)
    alignment = compute_normal_alignment(pcd)
    density = compute_density(mesh, pcd)
    points = np.asarray(pcd.points)
    # print(f'Mean sdf = {sdf.mean()}, Mean curv = {surf_variation.mean()}, Mean alignment = {alignment.mean()}, Mean density = {density.mean()}\n')

    full = np.column_stack((points, surf_variation, alignment, density))
    partial1 = resample_points(random_dropout(full), 1000)
    partial2 = resample_points(cut_by_sphere(full, bbox), 1000)

    return points, partial1, partial2


if __name__=="__main__":
    train_dir = './data/chairs/test'

    for chair_file in os.listdir(train_dir):
        print('Processing ', chair_file)

        chair_path = os.path.join(train_dir, chair_file)
        try:
            points, partial1, partial2 = process_mesh(chair_path)
        except Exception as e:
            print(f"Error occured: {e}")
            continue

        chair_name = os.path.splitext(chair_file)[0]
        out_dir = os.path.join('./data/chairs_processed/test', chair_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path_full = os.path.join(out_dir, 'full.npy')
        out_path_partial1 = os.path.join(out_dir, 'partial1.npy')
        out_path_partial2 = os.path.join(out_dir, 'partial2.npy')

        np.save(out_path_full, points)
        np.save(out_path_partial1, partial1)
        np.save(out_path_partial2, partial2)
        