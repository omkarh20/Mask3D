import torch
import numpy as np
import open3d as o3d
from mask3d import get_model, prepare_data
from pathlib import Path
from tkinter import Tk, filedialog

# Setup paths
CHECKPOINT_PATH = '/home/cave/3DReconstruction/Mask3D/mask3d/saved/scannet200_val.ckpt'

def select_input_file():
    """Open file dialog to select input PLY file."""
    root = Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select point cloud PLY file to segment",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    
    return file_path

def custom_save_colorized_mesh(mesh, labels, output_path):
    """Custom save function that colors mesh by predicted labels.
    
    Args:
        mesh: Open3D TriangleMesh with vertices
        labels: (N, 1) or (N,) array of class predictions
        output_path: Path to save colored PLY
    """
    # Flatten labels if needed
    if len(labels.shape) > 1:
        labels = labels.flatten()
    
    # Create a simple color map (cycle through colors)
    unique_labels = np.unique(labels)
    label_to_color = {}
    
    # Generate colors for each label
    np.random.seed(42)
    for label in unique_labels:
        label_to_color[label] = np.random.rand(3)
    
    # Assign colors to vertices
    colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        colors[i] = label_to_color[int(label)]
    
    # Create output point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved colored point cloud to {output_path}")
    print(f"Labels present: {unique_labels}")
    print(f"Label distribution: {[(l, (labels == l).sum()) for l in unique_labels]}")

def custom_map_output_to_pointcloud(num_points, outputs, inverse_map):
    """Custom mapping function that works with Mask3D output format.
    
    Args:
        num_points: Number of original points
        outputs: Model output dict with 'pred_logits' and 'pred_masks'
        inverse_map: Voxel to point mapping from prepare_data
    
    Returns:
        labels: (num_points,) array of semantic class predictions
    """
    pred_logits = outputs['pred_logits'][0]  # (150, 201) - [num_instances, num_classes]
    pred_masks = outputs['pred_masks']       # Could be list or tensor
    
    # Handle if pred_masks is a list
    if isinstance(pred_masks, list):
        pred_masks = pred_masks[0]
    
    print(f"\n=== CUSTOM MAP DEBUG ===")
    print(f"pred_masks type: {type(pred_masks)}")
    print(f"pred_masks shape: {pred_masks.shape}")
    print(f"inverse_map shape: {inverse_map.shape}")
    print(f"inverse_map range: [{inverse_map.min()}, {inverse_map.max()}]")
    print(f"========================\n")
    
    # Get semantic class for each instance (argmax over classes)
    instance_classes = pred_logits.argmax(dim=1)  # (150,) - class ID per instance
    
    # Get confidence scores for each instance
    instance_scores = pred_logits.softmax(dim=1).max(dim=1)[0]  # (150,)
    
    print(f"Instance classes range: [{instance_classes.min()}, {instance_classes.max()}]")
    print(f"Instance scores range: [{instance_scores.min():.3f}, {instance_scores.max():.3f}]")
    
    # Initialize label array for all voxels (all background = 0)
    num_voxels = pred_masks.shape[0]
    voxel_labels = np.zeros(num_voxels, dtype=np.int32)
    
    # Assign voxels to instances (in order of confidence)
    sorted_indices = np.argsort(-instance_scores.cpu().numpy())  # Sort by descending confidence
    
    for instance_idx in sorted_indices:
        # Get mask for this instance (column of pred_masks)
        mask = pred_masks[:, instance_idx].cpu().numpy() > 0.5  # (num_voxels,)
        class_id = instance_classes[instance_idx].item()
        voxel_labels[mask] = class_id
    
    # Map voxel labels back to original point cloud
    point_labels = voxel_labels[inverse_map]
    
    return point_labels

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
    
    # DEBUG: Check point scale and colors
    print(f"\n=== DEBUG INFO ===")
    print(f"Point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"            Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"            Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    room_size = points.max(axis=0) - points.min(axis=0)
    print(f"Room size: {room_size}")
    print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"Unique colors: {len(np.unique(colors, axis=0))}")
    print(f"=================\n")
    
    # Create mesh for full point cloud
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Prepare and infer on full cloud
    print("Preparing data (Voxelization)...")
    data, _, _, features, unique_map, inverse_map = prepare_data(mesh, device)
    print(f"  Voxelized to {data.C.shape[0]} occupied voxels")
    print(f"  Feature shape: {features.shape}")
    
    if data.C.shape[0] < 1000:
        print(f"  ⚠️  WARNING: Very few voxels ({data.C.shape[0]}) - point cloud might be too sparse!")
    
    print("Running Mask3D inference...")
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    print(f"\n=== MODEL OUTPUT DEBUG ===")
    if isinstance(outputs, dict):
        print(f"Output keys: {outputs.keys()}")
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    print(f"========================\n")
    
    print("Mapping predictions back to point cloud...")
    labels = custom_map_output_to_pointcloud(len(points), outputs, inverse_map)

    print(f"\n=== MAPPED LABELS DEBUG ===")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Labels range: [{labels.min()}, {labels.max()}]")
    print(f"Unique labels: {np.unique(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Top 10 classes: {list(zip(unique[-10:], counts[-10:]))}")
    print(f"========================\n")
    
    # Save colorized output
    print(f"Saving colorized output to {output_path}...")
    labels_reshaped = labels.reshape(-1, 1)
    custom_save_colorized_mesh(mesh, labels_reshaped, str(output_path))
    
    print("\n✓ Success! Inference complete.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
    main()
