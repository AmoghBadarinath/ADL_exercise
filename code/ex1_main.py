import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from my_models import ViT, CrossViT   
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cifar10"
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def early_stopping(val_acc, best_acc, patience_counter, patience):
    if val_acc > best_acc:
        # A new best model, reset counter
        best_acc = val_acc
        patience_counter = 0
    else:
        # No improvement, increment counter
        patience_counter += 1

    stop_training = patience_counter >= patience     
    return stop_training, best_acc, patience_counter

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    # Updated choices to include crossvit
    parser.add_argument('--model', type=str, default='r18', choices=['r18', 'vit', 'crossvit'], help='model to train (default: r18)') 
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization strength (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    return parser.parse_args()

def get_data_transforms():
    """
    Returns data transforms for training and testing, including augmentation for training.
    Uses CIFAR10 mean/std for better performance.
    """
    # Transforms for data augmentation on the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       
        transforms.RandomHorizontalFlip(),          
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Standard transforms for test/validation set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return train_transform, test_transform

def load_data(args):
    train_transform, test_transform = get_data_transforms()

    trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    
    # Split testset into validation (50%) and test (50%)
    val_size = len(testset) // 2
    test_size = len(testset) - val_size
    indices = torch.randperm(len(testset)).tolist()
    val_indices = indices[:val_size]
    test_indices = indices[val_size:]

    valloader = DataLoader(torch.utils.data.Subset(testset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(torch.utils.data.Subset(testset, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=2)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader

def get_model(args, device):
    if args.model == 'r18':
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10) 
    elif args.model == 'vit':
        # Optimized ViT parameters
        model = ViT(
            image_size = 32,
            patch_size = 4, # Small patch size
            num_classes = 10,
            dim = 128,      
            depth = 4,      
            heads = 4,      
            mlp_dim = 256,   
            dropout = 0.2,          
            emb_dropout = 0.2
        )
    elif args.model == 'crossvit':
        # CrossViT parameters
        model = CrossViT(
            image_size = 32, 
            num_classes = 10, 
            sm_dim = 64, 
            lg_dim = 128, 
            sm_patch_size = 4, 
            sm_enc_depth = 2, 
            sm_enc_heads = 8, 
            sm_mlp_dim = 128, 
            lg_patch_size = 8, 
            lg_enc_depth = 4, 
            lg_enc_heads = 8, 
            lg_mlp_dim = 256, 
            cross_attn_depth = 2, 
            cross_attn_heads = 8,
            dropout = 0.2, 
            emb_dropout = 0.2 
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    return model.to(device)

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    print(f"Epoch {epoch} [Train]: Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%")
    return avg_loss, train_accuracy

def test(model, device, test_loader, criterion, set="Test", epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    
    desc_str = f"Epoch {epoch} [{set}]" if epoch else f"[{set}]"
    pbar = tqdm(test_loader, desc=desc_str, leave=False)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    num_samples = len(test_loader.dataset)
    test_loss /= len(test_loader)
    accuracy = correct / num_samples
    
    print(f"{set} set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({100. * accuracy:.2f}%)")
    
    return accuracy, test_loss

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    trainloader, valloader, testloader = load_data(args)
    
    model = get_model(args, device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience = 50 # Increased patience for better Transformer training stability
    patience_counter = 0
    
    # CRITICAL FIX: Model-specific checkpoint path
    best_path = CKPT_DIR / f"best_{args.model}.pth" 
    os.makedirs("checkpoints", exist_ok=True)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        # 1. Training
        avg_train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, epoch)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # 2. Validation
        val_acc, avg_val_loss = test(model, device, valloader, criterion, set="Validation", epoch=epoch)
        val_accuracies.append(val_acc)
        val_losses.append(avg_val_loss)

        # Step the scheduler every epoch
        scheduler.step()

        # Check for early stopping and save best model
        stop_training, best_val_acc, patience_counter = early_stopping(val_acc, best_val_acc, patience_counter, patience)

        if val_acc == best_val_acc and patience_counter == 0 and args.save_model:
            # Only save if this is a new absolute best model
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved for {args.model} with validation accuracy: {best_val_acc*100:.2f}%")

        if stop_training:
            print(f"Early stopping triggered at epoch {epoch} due to no improvement in validation accuracy for {patience} epochs.")
            break
    
    # Final Evaluation on the Test Set
    print("\n" + "="*50)
    print("--- Final Test Set Evaluation ---")
    print("="*50)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"Loaded best {args.model} model for final test.")
    
    test_acc, _ = test(model, device, testloader, criterion, set="Test")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    # --- Save Training Graphs ---
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = f'{args.model}_training_performance_plots.png'
    plt.savefig(plot_path)
    print(f"\nTraining graphs saved successfully as {plot_path}")

if __name__ == '__main__':
    main()