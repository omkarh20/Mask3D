import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from pathlib import Path
from tkinter import Tk, filedialog

# Setup paths
NUM_REGIONS = 10  # Number of spatial regions to create

def select_input_file():
    """Open file dialog to select input PLY file."""
    root = Tk()
    root.withdraw()  # Hide the root window
    
    file_path = filedialog.askopenfilename(
        title="Select point cloud PLY file to split",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    
    return file_path

def split_spatial_regions(input_path: str, num_regions: int):
    """Split a point cloud into spatial regions using K-means clustering on coordinates.
    
    Each region will be a spatially isolated cluster. Points in each region are neighbors.
    
    Args:
        input_path: Path to input PLY file
        num_regions: Number of spatial regions to create
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: File {input_path} does not exist")
        return
    
    # Create output directory in same location as input
    output_dir = input_file.parent / f"{input_file.stem}_split"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading point cloud from {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Original cloud: {len(points)} points")
    
    # Cluster points spatially using K-means on XYZ coordinates
    print(f"\nClustering into {num_regions} spatial regions...")
    kmeans = KMeans(n_clusters=num_regions, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(points)
    
    print(f"Cluster centroids:\n{kmeans.cluster_centers_}")
    
    # Save each region
    for region_id in range(num_regions):
        mask = cluster_labels == region_id
        region_points = points[mask]
        region_colors = colors[mask]
        
        print(f"\n  Region {region_id+1}: {len(region_points)} points (centroid: {kmeans.cluster_centers_[region_id]})")
        
        # Create point cloud for this region
        region_pcd = o3d.geometry.PointCloud()
        region_pcd.points = o3d.utility.Vector3dVector(region_points)
        region_pcd.colors = o3d.utility.Vector3dVector(region_colors)
        
        # Save region
        output_file = output_dir / f'region_{region_id+1:02d}.ply'
        o3d.io.write_point_cloud(str(output_file), region_pcd)
        print(f"    ✓ Saved to {output_file}")
    
    print(f"\n✓ All {num_regions} regions saved to {output_dir}")

if __name__ == "__main__":
    input_file = select_input_file()
    if input_file:
        split_spatial_regions(input_file, NUM_REGIONS)
    else:
        print("No file selected. Exiting.")
