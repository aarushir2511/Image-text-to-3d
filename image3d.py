import os
import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from rembg import remove

def remove_background(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    result = remove(img)
    result.save(output_path)
    print(f"Background removed {output_path}")

#todo: dpeth est change
def depth_est(image_path):
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    img_rgba = Image.open(image_path).convert("RGBA")
    img_np = np.array(img_rgba)
    alpha_mask = img_np[:, :, 3] / 255.0
    img_rgb = img_np[:, :, :3]
    input_tensor = transform(img_rgb).to(device)

    print("[Depth Estimation")
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth *= alpha_mask
    max_depth = np.percentile(depth[depth > 0], 98)
    depth[depth > max_depth] = 0
    print("Depth map ready.")
    return depth

def create_point_cloud(depth_map, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    mask = z > 0
    x, y, z = x[mask], y[mask], z[mask]
    points = np.stack((x, y, z), axis=-1)

    colors = img_rgb[mask] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=5.0)
    print("Point cloud generated")
    return pcd

def reconstruct_mesh(pcd):
    pcd.estimate_normals()
    print("Reconstructing mesh")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print("Mesh reconstruction complete.")
    return mesh

def save_mesh(mesh, filename):
    if o3d.io.write_triangle_mesh(filename, mesh):
        print(f"Mesh saved → {filename}")
    else:
        print(f"Mesh save failed: {filename}")

def visualize_mesh(mesh):
    print("Mesh Visualization")
    o3d.visualization.draw_geometries([mesh])

def main():
    input_path = "input.png"
    no_bg_path = "subject.png"
    mesh_path = "output_mesh.obj"
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    remove_background(input_path, no_bg_path)
    depth_map = depth_est(no_bg_path)
    pcd = create_point_cloud(depth_map, no_bg_path)
    mesh = reconstruct_mesh(pcd)
    save_mesh(mesh, mesh_path)
    visualize_mesh(mesh)

if __name__ == "__main__":
    main()