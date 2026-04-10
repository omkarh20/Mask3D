import torch
import numpy as np
import open3d as o3d
from mask3d import get_model, prepare_data, map_output_to_pointcloud, save_colorized_mesh

# Setup paths
CHECKPOINT_PATH = '/home/cave/3DReconstruction/Mask3D/mask3d/saved/scannet200_val.ckpt'
INPUT_PLY_PATH = 'data/densified_iot2.ply'         # <--- CHANGE THIS to the path of your test .ply file
OUTPUT_PLY_PATH = 'data/densified_iot2_labelled2.ply' # Where the final colored mesh will be saved

# Chunking parameters
CHUNK_SIZE = 3.0  # Process 3m × 3m × 3m chunks
CHUNK_OVERLAP = 0.3  # 30cm overlap to prevent boundary artifacts

def chunk_point_cloud(points: np.ndarray, chunk_size: float, overlap: float) -> list:
    """Split point cloud into overlapping spatial chunks.
    
    Returns:
        List of tuples (point_indices, bounds) for each chunk
    """
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    chunks = []
    stride = chunk_size - overlap
    
    x = min_coords[0]
    while x < max_coords[0]:
        y = min_coords[1]
        while y < max_coords[1]:
            z = min_coords[2]
            while z < max_coords[2]:
                bounds = np.array([
                    [x, x + chunk_size],
                    [y, y + chunk_size],
                    [z, z + chunk_size]
                ])
                mask = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
                if mask.sum() > 100:  # Only keep non-empty chunks
                    chunks.append((np.where(mask)[0], bounds))
                z += stride
            y += stride
        x += stride
    
    print(f"Split into {len(chunks)} chunks")
    return chunks

def main():
    print("Loading Mask3D ScanNet200 model...")
    model = get_model(CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading point cloud from {INPUT_PLY_PATH}...")
    pcd = o3d.io.read_point_cloud(INPUT_PLY_PATH)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Original cloud: {len(points)} points")
    
    # Initialize label array
    labels = np.zeros(len(points), dtype=np.int32)
    
    # Split into chunks
    chunks = chunk_point_cloud(points, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Process each chunk
    for idx, (point_indices, bounds) in enumerate(chunks):
        print(f"\nProcessing chunk {idx+1}/{len(chunks)} ({len(point_indices)} points)...")
        
        # Create sub-mesh for this chunk (Mask3D expects an object with .vertices)
        chunk_mesh = o3d.geometry.TriangleMesh()
        chunk_mesh.vertices = o3d.utility.Vector3dVector(points[point_indices])
        chunk_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[point_indices])
        
        # Prepare and infer on chunk
        data, _, _, features, unique_map, inverse_map = prepare_data(chunk_mesh, device)
        print(f"  Voxelized to {data.C.shape[0]} occupied voxels")
        
        with torch.no_grad():
            outputs = model(data, raw_coordinates=features)
        
        # Map predictions back to chunk points
        chunk_labels = map_output_to_pointcloud(chunk_mesh, outputs, inverse_map)
        labels[point_indices] = chunk_labels.flatten() # Flatten from (N, 1) to (N,)
        print(f"  ✓ Chunk {idx+1} complete")
    
    # Save full prediction
    print(f"\nSaving colorized output to {OUTPUT_PLY_PATH}...")
    # Convert PointCloud to TriangleMesh for save_colorized_mesh compatibility
    output_mesh = o3d.geometry.TriangleMesh()
    output_mesh.vertices = o3d.utility.Vector3dVector(points)
    output_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # Reshape labels from (N,) to (N, 1) to match save_colorized_mesh expectations
    labels_reshaped = labels.reshape(-1, 1)
    save_colorized_mesh(output_mesh, labels_reshaped, OUTPUT_PLY_PATH, colormap='scannet200')
    
    print("\n✓ Success! Chunked inference complete.")
    print("You can now open the output file in a 3D viewer like MeshLab.")

if __name__ == "__main__":
    main()
