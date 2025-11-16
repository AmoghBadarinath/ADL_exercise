import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
# Import both models (assuming my_models.py is up-to-date)
from my_models import ViT, CrossViT 
from pathlib import Path
import os
import sys

# --- Configuration & Path Setup ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cifar10"
CKPT_DIR = BASE_DIR / "checkpoints"

# --- Model Parameters (MUST MATCH ex1_main.py) ---

# Parameters for the ViT (Matched to ex1_main.py)
ViT_PARAMS = dict(
    image_size = 32, patch_size = 4, num_classes = 10, dim = 128,
    depth = 4, heads = 4, mlp_dim = 256, dropout = 0.5, # Corrected dropout from 0.4 to 0.5
    emb_dropout = 0.5 # Corrected emb_dropout from 0.4 to 0.5
)

# Parameters for the CrossViT (Matched to ex1_main.py)
CrossViT_PARAMS = dict(
    image_size = 32, 
    num_classes = 10, 
    sm_dim = 64, 
    lg_dim = 128, 
    sm_patch_size = 4, # Corrected from 8 to 4
    sm_enc_depth = 2, 
    sm_enc_heads = 8, 
    sm_mlp_dim = 128, 
    lg_patch_size = 8, # Corrected from 16 to 8
    lg_enc_depth = 4, # Corrected from 2 to 4
    lg_enc_heads = 8, 
    lg_mlp_dim = 256, # Corrected from 128 to 256
    cross_attn_depth = 2, 
    cross_attn_heads = 8,
    dropout = 0.3, # Corrected from 0.1 to 0.3
    emb_dropout = 0.3 # Corrected from 0.1 to 0.3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The image index to visualize (can be changed)
IMAGE_INDEX = 15

# --- Data Loading and Transformation ---
# Using the same normalization as in ex1_main.py
test_transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])

try:
    test_dataset = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_transform_norm)
except Exception as e:
    print(f"Could not load CIFAR10 dataset: {e}")
    sys.exit(1)

# --- Visualization Function ---
def visualize_attention(model_name, checkpoint_path):
    
    # 1. Load Model based on selection
    if model_name == 'vit':
        model_params = ViT_PARAMS
        model_class = ViT
        P = model_params['patch_size'] # Patch size for ViT is 4x4
        print("Loading ViT model...")
    elif model_name == 'crossvit':
        model_params = CrossViT_PARAMS
        model_class = CrossViT
        P = model_params['sm_patch_size'] # Small patch size for CrossViT is 4x4
        print("Loading CrossViT model...")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Initialize model
    model = model_class(**model_params)
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}.")
        print("Please ensure the checkpoint is in the 'checkpoints' directory.")
        return

    # Load the state dictionary
    # NOTE: map_location=device ensures it loads correctly onto the current device.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"{model_name.upper()} model loaded successfully.")

    # 2. Prepare Image
    normalized_img, label_idx = test_dataset[IMAGE_INDEX]
    class_name = test_dataset.classes[label_idx]
    
    # Denormalize the image for display (CIFAR-10 raw data is in range 0-255)
    # The raw_img from test_dataset.data is already in 0-255 uint8 format
    raw_img = test_dataset.data[IMAGE_INDEX] 

    normalized_img = normalized_img.unsqueeze(0).to(device)

    # 3. Perform Forward Pass and Extract Weights
    
    # CRITICAL: We need to modify the models to save attention weights
    # Since my_models.py doesn't have an easy way to get the last block's weights
    # for CrossViT (due to nesting), we must apply the fix from the previous step 
    # and assume the user's my_models.py is fixed to allow this.
    try:
        model.set_save_attention_weights(True)
    except AttributeError:
        # If the function is not found, it means my_models.py is not updated
        print("\nFATAL ERROR: The 'my_models.py' file must be updated with 'set_save_attention_weights' functionality.")
        print("Please check previous instructions to ensure your 'my_models.py' is the final, corrected version.")
        return
        
    with torch.no_grad():
        output = model(normalized_img) 
        
        # INFERENCE: Calculate prediction and confidence
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item() * 100
        
        is_correct = (predicted_index == label_idx)
        prediction_status = "CORRECT" if is_correct else "INCORRECT"

        print(f"\n--- INFERENCE RESULT ---")
        print(f"Image Index: {IMAGE_INDEX}")
        print(f"Prediction Status: {prediction_status}")
        print(f"True Class: {class_name}")
        print(f"Predicted Class: {test_dataset.classes[predicted_index]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"------------------------")

        # Retrieve weights from the last block
        try:
            # .squeeze(0) removes the batch dimension -> (H, N_tokens, N_tokens)
            attn_weights = model.get_last_attention_weights().squeeze(0) 
        except AttributeError as e:
            print(f"Error getting attention weights: {e}")
            print("Ensure 'my_models.py' is updated with 'get_last_attention_weights'.")
            return

    # The first token (index 0) is the CLS token. We look at its attention to all patches (index 1 and onward).
    # Shape of cls_to_patches_attn: (H, N_patches)
    cls_to_patches_attn = attn_weights[:, 0, 1:] 
    
    # Average attention across all heads (H) for a single map
    average_attn = cls_to_patches_attn.mean(dim=0).cpu().numpy()

    # 4. Reshape Attention Weights into a 2D Map 
    # For ViT and the small stream of CrossViT, Image size is 32. Patch size P is 4.
    I = model_params['image_size']    
    grid_size = I // P 

    if average_attn.size != grid_size * grid_size:
        print(f"\nError: Mismatch in attention map size.")
        print(f"Expected {grid_size * grid_size} patch tokens, but got {average_attn.size} tokens.")
        return

    attention_map = average_attn.reshape(grid_size, grid_size)
    
    # Scale up the attention map to the full image size for visualization
    attention_map = np.kron(attention_map, np.ones((P, P))) 

    # 5. Plotting
    plt.figure(figsize=(8, 4))
    
    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.title(f"Original: {class_name}")
    plt.axis('off')
    
    # Subplot 2: Heatmap Overlay 
    plt.subplot(1, 2, 2)
    plt.imshow(raw_img, alpha=0.5) 
    plt.imshow(attention_map, cmap='jet', alpha=0.5, interpolation='nearest') 
    plt.colorbar(label='CLS-to-Patch Attention Score')
    plt.title(f"{model_name.upper()} Attention (Last Block)")
    plt.axis('off')
    
    plt.tight_layout()
    output_filename = f"{model_name}_attention_visualization.png"
    plt.savefig(output_filename)
    print(f"\n✅ Attention visualization saved as '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize ViT/CrossViT Attention')
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'crossvit'],
                        help='Model to visualize (vit or crossvit)')
    args = parser.parse_args()

    # Map the argparse choice to the user's actual uploaded filenames
    checkpoint_paths = {
        # The training script saves ViT as 'vit_best_model.pth' and we assume CrossViT as 'cvit_best_model.pth'
        'vit': CKPT_DIR / "vit_best_model.pth", 
        'crossvit': CKPT_DIR / "cvit_best_model.pth" 
    }
    
    visualize_attention(args.model, checkpoint_paths[args.model])