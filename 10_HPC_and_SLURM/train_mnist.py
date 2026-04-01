"""
MNIST Training Script for HPC/SLURM
Machine Learning 101 - Module 10

This is the same neural network from Module 08, adapted to run as a
standalone script on an HPC cluster. Key changes from the notebook:
  - Command-line arguments via argparse
  - Proper logging (instead of print statements)
  - Model checkpoint saving
  - No matplotlib (no display available in batch jobs)

Usage:
    python train_mnist.py --epochs 10 --lr 0.001 --batch-size 64
"""

import argparse
import logging
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save model and logs")
    return parser.parse_args()


class NeuralNetwork(nn.Module):
    """Same architecture from Module 08 - 3 layer feedforward network."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * X.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
        ],
    )
    logger = logging.getLogger(__name__)

    # Log job info
    logger.info("MNIST Training Script - Machine Learning 101")
    logger.info(f"Arguments: {vars(args)}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    slurm_node = os.environ.get("SLURMD_NODENAME", "local")
    logger.info(f"SLURM Job ID: {slurm_job_id}")
    logger.info(f"SLURM Node: {slurm_node}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data
    data_dir = os.environ.get("SLURM_TMPDIR", "./data")
    logger.info(f"Downloading data to: {data_dir}")

    training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=ToTensor())

    logger.info(f"Training set: {len(training_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")

    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = NeuralNetwork().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.1f}% - "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.1f}%"
        )

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": test_acc,
                "test_loss": test_loss,
            }, model_path)
            logger.info(f"Saved best model (acc: {test_acc * 100:.1f}%) to {model_path}")

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Best test accuracy: {best_acc * 100:.1f}%")


if __name__ == "__main__":
    main()
