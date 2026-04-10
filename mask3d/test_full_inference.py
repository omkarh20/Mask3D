import torch
import numpy as np
import open3d as o3d
from mask3d import get_model, prepare_data, map_output_to_pointcloud, save_colorized_mesh
from pathlib import Path
from tkinter import Tk, filedialog

# Setup paths
CHECKPOINT_PATH = '/home/cave/3DReconstruction/Mask3D/mask3d/saved/scannet200_val.ckpt'

def select_input_file():
    """Open file dialog to select input PLY file."""
    root = Tk()
    root.withdraw()  # Hide the root window
    
    file_path = filedialog.askopenfilename(
        title="Select point cloud PLY file to segment",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    
    return file_path

def main():
    # Select input file
    input_path = select_input_file()
    if not input_path:
        print("No file selected. Exiting.")
        return
    
    input_file = Path(input_path)
    
    # Generate output path with same name + _labelled
    output_path = input_file.parent / f"{input_file.stem}_labelled.ply"
    
    print("Loading Mask3D ScanNet200 model...")
    model = get_model(CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading point cloud from {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"Original cloud: {len(points)} points")
    
    # Create mesh for full point cloud
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Prepare and infer on full cloud
    print("Preparing data (Voxelization)...")
    data, _, _, features, unique_map, inverse_map = prepare_data(mesh, device)
    print(f"  Voxelized to {data.C.shape[0]} occupied voxels")
    
    print("Running Mask3D inference...")
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    
    print("Mapping predictions back to point cloud...")
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)
    
    # Save colorized output
    print(f"Saving colorized output to {output_path}...")
    labels_reshaped = labels.flatten().reshape(-1, 1)
    save_colorized_mesh(mesh, labels_reshaped, str(output_path), colormap='scannet200')
    
    print("\n✓ Success! Inference complete.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
