import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Model is loading.")
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))


prompt = "a brown cap"

print(f"Generating 3D model")
latents = sample_latents(
    model=model,
    diffusion=diffusion,
    batch_size=1,
    model_kwargs={"texts": [prompt]},
    guidance_scale=15.0,
    device=device,
    progress=True
)

def convert_trimesh(shap_e_mesh):
    vertices = shap_e_mesh.verts
    faces = shap_e_mesh.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

for i, latent in enumerate(latents):
    print(f"Processing latent {i+1}/{len(latents)}...")
    shap_e_mesh = decode_latent_mesh(latent).tri_mesh()
    
    mesh_path = f"{prompt.replace(' ', '_')}_mesh.obj"
    print(f"Saving mesh to {mesh_path}...")
    with open(mesh_path, "w") as f:
        shap_e_mesh.write_obj(f)
    print(f"Mesh saved to {mesh_path}")
    
    # Convert to trimesh and visualize
    try:
        mesh = convert_trimesh(shap_e_mesh)

        print("Visualization")
        scene = trimesh.Scene(mesh)
        
        png_path = f"{prompt.replace(' ', '_')}_preview.png"
        print(f"Saving preview image to {png_path}")
        scene.save_image(resolution=[1024, 768], path=png_path)
        
        print("Opening interactive viewer")
        scene.show()
    except Exception as e:
        print(f"Error in visualization: {e}")

print("Process completed successfully!")