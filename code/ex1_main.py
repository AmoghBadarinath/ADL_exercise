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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cifar10"
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def early_stopping(val_acc, best_acc, patience_counter, patience):
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
    else:
        patience_counter += 1

    stop_training = patience_counter >= patience     
    return stop_training, best_acc, patience_counter

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.003)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='AdamW weight decay (default: 1e-2)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    val_acc = correct / len(test_loader.dataset)
    return val_acc

def run(args):

    # Download and load the training data
    train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ) 
                                    ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                      # ImageNet mean/std values should also fit okayish for CIFAR
                                      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ) 
                                      ])

    dataset = datasets.CIFAR10(DATA_DIR, download=True, train=True, transform=train_transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10(DATA_DIR, download=True, train=False, transform=test_transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    # Use AdamW optimizer and a CosineAnnealingLR scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience = 50
    patience_counter = 0
    best_path = CKPT_DIR / "best_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        val_acc = test(model, device, valloader, criterion, set="Validation")

        # Step the scheduler every epoch
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved with validation accuracy: {best_val_acc*100:.2f}%")

        stop_training, best_val_acc, patience_counter = early_stopping(val_acc, best_val_acc, patience_counter, patience)

        if stop_training:

            print("Early stopping triggered at epoch {}".format(epoch))

            break

    if args.save_model:
        # Load the best model for testing
        model.load_state_dict(torch.load(best_path))
        print("Loaded best model for testing.")
    test(model, device, testloader, criterion)

if __name__ == '__main__':
    args = parse_args()
    run(args)
