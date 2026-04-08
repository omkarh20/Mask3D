import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R_scipy

def power_densify_gs(input_path, detail_samples=30, base_samples=5):
    print("Loading PLY...")
    plydata = PlyData.read(input_path)
    v = plydata['vertex']
    
    # 1. Filter by Opacity (Lowered to 0.3 to preserve door frames and thin edges)
    opacity = 1 / (1 + np.exp(-v['opacity']))
    mask = opacity > 0.3
    
    # 2. Extract and filter attributes
    x = v['x'][mask]
    y = v['y'][mask]
    z = v['z'][mask]
    mu = np.vstack((x, y, z)).T  # Shape: (N, 3)
    
    scale_names = ['scale_0', 'scale_1', 'scale_2']
    scales = np.exp(np.vstack([v[s][mask] for s in scale_names]).T) # Shape: (N, 3)
    
    rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    quats_wxyz = np.vstack([v[r][mask] for r in rot_names]).T
    quats_xyzw = np.roll(quats_wxyz, shift=-1, axis=1) 
    rot_matrices = R_scipy.from_quat(quats_xyzw).as_matrix() # Shape: (N, 3, 3)
    
    dc_names = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    colors = 1 / (1 + np.exp(np.vstack([v[c][mask] for c in dc_names]).T)) # Shape: (N, 3)
    
    N = mu.shape[0]
    print(f"Processing {N} valid Gaussians...")

    # 3. Dynamic Sampling Logic
    # Calculate the maximum scale of each Gaussian to determine if it's a detail or a wall
    scale_max = np.max(scales, axis=1)
    
    # If the max scale is very small (< 0.05), it's likely an intricate object. Give it high density.
    sample_counts = np.where(scale_max < 0.05, detail_samples, base_samples)
    total_points = np.sum(sample_counts)
    
    print(f"Generating {total_points} total points using dynamic sampling...")

    # 4. Vectorized Memory Expansion (The fast way to handle variable samples)
    repeated_mu = np.repeat(mu, sample_counts, axis=0)
    repeated_scales = np.repeat(scales, sample_counts, axis=0)
    repeated_rot_matrices = np.repeat(rot_matrices, sample_counts, axis=0)
    repeated_colors = np.repeat(colors, sample_counts, axis=0)

    # 5. Monte Carlo Sampling
    # Generate all random noise at once
    random_samples = np.random.normal(0, 1, (total_points, 3))
    
    # Apply scales
    scaled_samples = random_samples * repeated_scales
    
    # Apply rotations (Batched Matrix-Vector multiplication)
    rotated_samples = np.einsum('nij,nj->ni', repeated_rot_matrices, scaled_samples)
    
    # Translate by mean
    all_points = rotated_samples + repeated_mu

    # 6. Export to Open3D and Estimate Normals
    print("Building Open3D point cloud and estimating normals...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(repeated_colors)
    
    # Normal estimation gives Mask3D extra geometric context for edge detection
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Orient normals consistently (optional, but helps some models)
    #pcd.orient_normals_consistent_tangent_plane(100)

    return pcd

# Usage
pcd = power_densify_gs("data/iot_splat.ply", detail_samples=15, base_samples=3)
o3d.io.write_point_cloud("data/densified_iot2.ply", pcd)
print("Saved densified point cloud successfully!")
