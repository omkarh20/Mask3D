import open3d as o3d
import numpy as np
from plyfile import PlyData

def convert_gs_to_standard_ply(input_path, output_path):
    plydata = PlyData.read(input_path)
    v = plydata['vertex']
    
    # Extract XYZ
    points = np.stack([v['x'], v['y'], v['z']], axis=1)
    
    # Extract Base Color (DC component of Spherical Harmonics)
    # 3DGS colors are stored in log-space/SH. This is a rough approximation:
    if 'f_dc_0' in v:
        rgbs = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
        # Simple sigmoid-ish mapping to get color back to 0-1
        rgbs = 1 / (1 + np.exp(-rgbs)) 
    else:
        # Fallback if it already has standard colors
        rgbs = np.stack([v['red'], v['green'], v['blue']], axis=1) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Clean point cloud saved to {output_path}")

convert_gs_to_standard_ply('data/point_cloud.ply', 'data/point_cloud_clean.ply')
