import torch
from mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh

# 1. Setup paths (Make sure your new ScanNet200 checkpoint is here!)
CHECKPOINT_PATH = '/home/cave/3DReconstruction/Mask3D/mask3d/saved/scannet200_val.ckpt'
INPUT_PLY_PATH = 'data/densified_iot2.ply'         # <--- CHANGE THIS to the path of your test .ply file
OUTPUT_PLY_PATH = 'data/densified_iot2_labelled.ply' # Where the final colored mesh will be saved

def main():
    print("Loading Mask3D ScanNet200 model...")
    # 2. Load the model and move to your RTX 40-series GPU
    model = get_model(CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading 3D point cloud from {INPUT_PLY_PATH}...")
    # 3. Load and prepare data (Voxelization)
    mesh = load_mesh(INPUT_PLY_PATH)
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)

    print("Running Mask3D inference... (This might take a moment)")
    # 4. Run inference (No gradients needed)
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        
    print("Mapping predictions back to the original point cloud...")
    # 5. Map the chunks back to the original high-res mesh
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)

    print(f"Saving colorized output to {OUTPUT_PLY_PATH}...")
    # 6. Save the output using the expanded 200-class scannet colormap
    save_colorized_mesh(mesh, labels, OUTPUT_PLY_PATH, colormap='scannet200')

    print("\nSuccess! You can now open the output file in a 3D viewer like MeshLab.")

if __name__ == "__main__":
    main()
