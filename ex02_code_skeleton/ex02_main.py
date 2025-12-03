import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from PIL import Image
from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from datasets import load_dataset  # For Exercise 2.5
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--cfg', action='store_true', default=False, help='Enable Classifier-Free Guidance')
    parser.add_argument('--p_uncond', type=float, default=0.1, help='Prob. of uncond. training')
    return parser.parse_args()

def plot_metrics(train_losses, val_losses, store_path, run_name):
    """Generates and saves plots for both Loss and Proxy Accuracy."""
    
    # Ensure save directory exists
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    epochs = range(1, len(train_losses) + 1)
    
    # --- 1. Loss Plot (MSE) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss (MSE)', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss (MSE)', marker='o')
    
    plt.title(f'Training and Validation Loss for {run_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(store_path, "loss_history.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}")
    
    # --- 2. Accuracy Plot (Inverse Loss Proxy) ---
    # Proxy Accuracy = 1 / Loss (as higher is better)
    train_proxy_acc = [1 / l for l in train_losses]
    val_proxy_acc = [1 / l for l in val_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_proxy_acc, label='Training Proxy Accuracy ($1/\mathrm{Loss}$)', marker='x')
    plt.plot(epochs, val_proxy_acc, label='Validation Proxy Accuracy ($1/\mathrm{Loss}$)', marker='x')
    
    plt.title(f'Training and Validation Accuracy Proxy for {run_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Proxy Accuracy ($1/\mathrm{Loss}$)')
    plt.legend()
    plt.grid(True)
    
    acc_plot_path = os.path.join(store_path, "accuracy_proxy.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Saved accuracy proxy plot to {acc_plot_path}")

def sample_and_save_images(n_images, diffusor, model, device, store_path, cfg=False, num_classes=10):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # Sample standard (or with CFG if enabled)
    # Let's sample a few classes if CFG is on
    y = None
    if cfg:
        # Create labels 0 to n_images-1 (looping if n_images > num_classes)
        y = torch.tensor([i % num_classes for i in range(n_images)], device=device).long()
        print(f"Sampling with CFG, classes: {y.tolist()}")
    
    # Sample
    final_images, intermediates = diffusor.sample(
        model, 
        image_size=diffusor.img_size, 
        batch_size=n_images, 
        channels=3, 
        y=y, 
        cfg_scale=3.0 if cfg else 0.0, 
        return_all_timesteps=True
    )
    
    # Helper to process tensor to nice image format (0-255 uint8)
    def process_image(img_tensor):
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
        img_tensor = (img_tensor * 255).type(torch.uint8)
        return img_tensor

    # --- A. Save Final Grid ---
    # [cite_start]Citation: quantitatively assess the quality of your images [cite: 94]
    final_grid = make_grid(process_image(final_images).float() / 255.0, nrow=4)
    save_image(final_grid, os.path.join(store_path, "final_samples.png"))

    # --- B. Save Evolution of a Single Sample (Static Grid) ---
    # We pick the first image in the batch and show it at distinct timesteps
    # e.g., t=Start, t=75%, t=50%, t=25%, t=End
    idx_to_show = 0 
    num_steps = len(intermediates)
    # Select ~8 evenly spaced timesteps
    indices = np.linspace(0, num_steps - 1, 8, dtype=int)
    
    evolution_frames = [intermediates[i][idx_to_show] for i in indices]
    evolution_tensor = torch.stack(evolution_frames)
    
    evo_grid = make_grid(process_image(evolution_tensor).float() / 255.0, nrow=8)
    save_image(evo_grid, os.path.join(store_path, "evolution_process.png"))
    print(f"Saved visualization artifacts to {store_path}")

    # --- C. Create GIF Animation ---
    # [cite_start]Citation: Create a couple of animations [cite: 97]
    # We will animate the first 16 images in the batch as a grid
    
    frames = []
    # Downsample frames to keep GIF size reasonable (e.g., every 2nd or 5th frame)
    step_size = max(1, len(intermediates) // 50) 
    
    print("Generating GIF...")
    for i in range(0, len(intermediates), step_size):
        # Create a grid for this timestep
        batch_grid = make_grid(process_image(intermediates[i]).float() / 255.0, nrow=4, padding=2)
        # Convert to PIL Image
        ndarr = batch_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        frames.append(im)
        
    # Save GIF
    frames[0].save(
        os.path.join(store_path, "diffusion_movie.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100, # Duration of each frame in ms
        loop=0
    )
    print("GIF saved.")




def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()
        total_loss = 0

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % args.log_interval == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        if args.dry_run:
            break
    avg_train_loss = total_loss / len(trainloader)
    print(f"Training Epoch {epoch} Average Loss: {avg_train_loss:.6f}")
    return avg_train_loss

def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(testloader):
            images = images.to(device)
            t = torch.randint(0, args.timesteps, (len(images),), device=device).long()
            
            # Compute loss on test set
            loss = diffusor.p_losses(model, images, t, loss_type="l2")
            total_loss += loss.item()
            
    avg_loss = total_loss / len(testloader)
    print(f"Test Set Average Loss: {avg_loss:.6f}")
    return avg_loss

def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,), class_free_guidance=args.cfg,
        num_classes=10).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.run_name == "DDPM_Linear":
        my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    elif args.run_name == "DDPM_Cosine" or "DDPM_CFG":     
        my_scheduler = lambda x: cosine_beta_schedule(x)

    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('./data/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('./datagit/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # --- Loss Tracking Lists ---
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = Path("results") / args.run_name # Creates results/DDPM_Linear
    save_path.mkdir(parents=True, exist_ok=True)
    n_images = 8

    # --- PLOT LOSSES ---
    plot_metrics(train_losses, val_losses, save_path, args.run_name)

    sample_and_save_images(n_images, diffusor, model, device, save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    run(args)
