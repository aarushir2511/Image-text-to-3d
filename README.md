# Image to 3D Model

This project takes a 2D input image and generates a 3D `.obj` mesh model using a combination of background removal, depth estimation, and 3D reconstruction techniques. It leverages PyTorch, MiDaS, Open3D, and rembg to create visually accurate point clouds and meshes.

---

## Thought Process

The goal was to generate a 3D object from a simple image. Here's the logic:

1. **Remove Background**: Makes the subject more distinct for point cloud creation.
2. **Depth Estimation**: Use MiDaS to convert RGB to a depth map.
3. **Point Cloud Creation**: Combine RGB + depth + alpha mask to make a 3D point cloud.
4. **Mesh Reconstruction**: Convert point cloud into a 3D mesh using Poisson surface reconstruction.
5. **Output & Visualize**: Save the mesh as `.obj` and visualize it.

---

## Libraries Used

- `torch` – PyTorch for MiDaS model
- `opencv-python` – Image processing
- `numpy` – Numerical operations
- `open3d` – Point cloud and mesh processing
- `Pillow` – Image handling
- `rembg` – Background removal
- `MiDaS` – Depth estimation from single image (via torch.hub)

---

# Text to 3D Model

This script generates a 3D model from a text prompt using OpenAI’s Shape-E diffusion model and visualizes it.

## Libraries Used

- **torch** – For running the deep learning model.  
- **numpy** – For numerical operations.  
- **trimesh** – For 3D mesh handling and visualization.  
- **shap_e** – OpenAI’s text-to-3D generation model.  

## Thought Process

- Take a text prompt as input.  
- Generate a 3D object using Shape-E.  
- Save the object as a `.obj` file.  
- Create a preview image using `trimesh`.  
- Launch an interactive viewer for quick inspection.


## Steps to Run
**Install dependencies**  
   Shape-E and all necessary libraries are already listed in `requirements.txt`, so just run it, given below.


## Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt  (includes shaoe-e in req file)

#run
python image3d.py
python text3d.py
