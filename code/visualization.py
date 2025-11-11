import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
# Assuming my_models.py contains the CrossViT class
# NOTE: The CrossViT implementation is not shown, but we must
# assume it has a method `get_last_attention_weights()`.
from my_models import CrossViT 
from pathlib import Path
import os
import sys

# --- Configuration & Path Setup ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cifar10"
CKPT_DIR = BASE_DIR / "checkpoints"
best_path = CKPT_DIR / "best_model.pth"

# Use the detailed, correct parameters for CrossViT (inferred from your original training script)
# The key difference is using sm_patch_size and lg_patch_size instead of just patch_size.
CrossViT_PARAMS = dict(
    image_size = 32, 
    num_classes = 10, 
    # Small branch parameters (Patch size 8x8)
    sm_dim = 64, 
    sm_patch_size = 8, 
    sm_enc_depth = 2,
    sm_enc_heads = 8, 
    sm_enc_mlp_dim = 128, 
    sm_enc_dim_head = 64,
    # Large branch parameters (Patch size 16x16) - Assumed 16 since image_size=32 and sm_patch_size=8
    lg_dim = 128, 
    lg_patch_size = 16, 
    lg_enc_depth = 2, 
    lg_enc_heads = 8, 
    lg_enc_mlp_dim = 128, 
    lg_enc_dim_head = 64,
    # Cross-attention parameters
    cross_attn_depth = 2, 
    cross_attn_heads = 8,
    cross_attn_dim_head = 64, 
    # General parameters
    depth = 3, 
    dropout = 0.1,
    emb_dropout = 0.1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The image index to visualize 
IMAGE_INDEX = 10 

# --- Data Loading and Transformation ---
# NOTE: Using the CIFAR10 mean/std for accurate data loading (different from ImageNet ones in your training)
# You must ensure this matches the `test_transform` used during the training that saved the checkpoint!
test_transform_norm = transforms.Compose([
    transforms.ToTensor(),
    # These are standard CIFAR10 mean/std, recommended for consistency.
    # If your model was trained with ImageNet mean/std, use those instead: (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])

# Use the correct data directory
try:
    test_dataset = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_transform_norm)
except Exception as e:
    print(f"Could not load CIFAR10 dataset: {e}")
    sys.exit(1)

# --- Visualization Function ---
def visualize_attention():
    # 1. Load Model
    # CrossViT_PARAMS is now correctly structured for CrossViT constructor
    model = CrossViT(**CrossViT_PARAMS)
    
    if not best_path.exists():
        print(f"Error: Model checkpoint not found at {best_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    print("CrossViT model loaded successfully.")

    # 2. Prepare Image
    # Get the normalized image for the model forward pass and the raw image for visualization
    normalized_img, label_idx = test_dataset[IMAGE_INDEX]
    
    # Get the class name for the title
    class_name = test_dataset.classes[label_idx]

    # Get the original, un-normalized image (NumPy array HxWxC)
    # NOTE: It's important to get the data *before* the normalization transform for visualization.
    raw_img = test_dataset.data[IMAGE_INDEX] 

    # Add batch dimension and move to device
    normalized_img = normalized_img.unsqueeze(0).to(device)

    # 3. Perform Forward Pass and Extract Weights
    with torch.no_grad():
        # The forward pass must compute and store the attention weights internally
        _ = model(normalized_img) 
        
        # Retrieve weights from the last block. 
        # ASSUMPTION: This method returns the attention of the FINAL CLS token 
        # to ALL tokens from the small branch or combined tokens, shape (Heads, N_tokens, N_tokens)
        try:
            attn_weights = model.get_last_attention_weights().squeeze(0) # (H, N_tokens, N_tokens)
        except AttributeError:
            print("Error: `get_last_attention_weights()` method not found or failed.")
            print("Ensure your `CrossViT` class in `my_models.py` correctly implements this method.")
            return

    # The first token (index 0) is the CLS token. We look at its attention to all patches (index 1 and onward).
    cls_to_patches_attn = attn_weights[:, 0, 1:] 
    
    # Average attention across all heads (H) for a single map
    # Shape: (N_patches)
    average_attn = cls_to_patches_attn.mean(dim=0).cpu().numpy()

    # 4. Reshape Attention Weights into a 2D Map (Using Small Patch Size for finer detail)
    P = CrossViT_PARAMS['sm_patch_size'] # Use the small patch size (8) for resolution
    I = CrossViT_PARAMS['image_size']   # Image size (32)
    
    # N_patches from small branch = (Image_size / sm_patch_size) ** 2 = (32 / 8)**2 = 16
    grid_size = I // P # e.g., 32 // 8 = 4

    # The attention must be reshaped to the grid size (4x4)
    if average_attn.size != grid_size * grid_size:
        print(f"\nWarning: Expected {grid_size * grid_size} attention values for a {grid_size}x{grid_size} grid, but got {average_attn.size}.")
        print("This usually means the `get_last_attention_weights` method returned attention over the LARGER patches or combined tokens.")
        # Attempt to reshape using the grid size of the large patches for robustness
        lg_grid_size = I // CrossViT_PARAMS['lg_patch_size'] # 32//16 = 2
        if average_attn.size == lg_grid_size * lg_grid_size:
            print(f"Reshaping to Large Patch Grid Size ({lg_grid_size}x{lg_grid_size}).")
            grid_size = lg_grid_size
            P = CrossViT_PARAMS['lg_patch_size']
        else:
            print("Cannot reshape. Visualization aborted.")
            return

    attention_map = average_attn.reshape(grid_size, grid_size)
    
    # Scale up the attention map to the full image size for visualization
    # Scales the GxG grid up to 32x32
    attention_map = np.kron(attention_map, np.ones((P, P))) 

    # 5. Plotting
    plt.figure(figsize=(8, 4))
    
    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.title(f"Original: {class_name} (Index: {IMAGE_INDEX})")
    plt.axis('off')
    
    # Subplot 2: Heatmap Overlay 
    plt.subplot(1, 2, 2)
    plt.imshow(raw_img, alpha=0.5) # Show image with some transparency
    # Overlay the attention heatmap
    plt.imshow(attention_map, cmap='jet', alpha=0.5, interpolation='nearest') 
    plt.colorbar(label='CLS-to-Patch Attention Score')
    plt.title("Attention Map (Last Block CLS Token)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_attention()