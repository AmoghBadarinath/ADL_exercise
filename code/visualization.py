import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from my_models import ViT
from pathlib import Path

# --- Configuration ---
CKPT_DIR = Path("./checkpoints")
best_path = CKPT_DIR / "best_model.pth"
# Use the same parameters you used for training ViT
VIT_PARAMS = dict(
    image_size=32, patch_size=8, num_classes=10, dim=64, depth=2, heads=8, mlp_dim=128,
    dropout=0.1, emb_dropout=0.1
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The image index to visualize (e.g., first image in the test set)
IMAGE_INDEX = 10 

# --- Data Loading and Transformation ---
# We use the test set transformation, but we need an un-normalized image for visualization
test_transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform_norm)

# --- Model Loading and Extraction ---
def visualize_attention():
    # 1. Load Model
    model = ViT(**VIT_PARAMS)
    if not best_path.exists():
        print(f"Error: Model checkpoint not found at {best_path}. Please train the ViT model first using 'python ex1_main.py --model vit --save-model'")
        return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    print("ViT model loaded successfully.")

    # 2. Prepare Image
    # Get the normalized image for the model forward pass and the raw image for visualization
    normalized_img, label = test_dataset[IMAGE_INDEX]
    
    # We need the original, un-normalized image to overlay the heatmap on
    raw_img = test_dataset.data[IMAGE_INDEX] # NumPy array (32, 32, 3)

    # Add batch dimension and move to device
    normalized_img = normalized_img.unsqueeze(0).to(device)

    # 3. Perform Forward Pass and Extract Weights
    with torch.no_grad():
        # The forward pass populates the self.attn_weights attribute
        _ = model(normalized_img) 
        
        # Retrieve weights from the last block
        attn_weights = model.get_last_attention_weights().squeeze(0) # (H, N_tokens, N_tokens)
        
    # The first token (index 0) is the CLS token. We want its attention to all other tokens (patches).
    # Shape: (H, 1, N_patches + 1) -> (H, N_patches + 1)
    cls_to_patches_attn = attn_weights[:, 0, 1:] 
    
    # Average attention across all heads (H) for a single map
    # Shape: (N_patches)
    average_attn = cls_to_patches_attn.mean(dim=0).cpu().numpy()

    # 4. Reshape Attention Weights into a 2D Map
    # N_patches = (Image_size / Patch_size) ** 2
    P = VIT_PARAMS['patch_size']
    I = VIT_PARAMS['image_size']
    grid_size = I // P # e.g., 32 // 8 = 4

    # Reshape the 1D attention vector (e.g., 16 values) into a 2D grid (4x4)
    attention_map = average_attn.reshape(grid_size, grid_size)
    
    # Scale up the attention map to the full image size for visualization
    attention_map = np.kron(attention_map, np.ones((P, P))) # e.g., scale 4x4 to 32x32

    # 5. Plotting
    plt.figure(figsize=(8, 4))
    
    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.title(f"Original Image (Label: {label})")
    plt.axis('off')
    
    # Subplot 2: Heatmap Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(raw_img, alpha=0.5) # Show image with some transparency
    # Overlay the attention heatmap
    # Use 'jet' colormap for clear visualization, set interpolation for smooth edges
    plt.imshow(attention_map, cmap='jet', alpha=0.5, interpolation='nearest') 
    plt.colorbar(label='CLS-to-Patch Attention Score')
    plt.title("Attention Map (Last Block CLS Token)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_attention()