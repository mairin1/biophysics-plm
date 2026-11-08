import numpy as np
import pandas as pd

def load_splits():
    import os
    # Load split indices (taken from METL)
    splits = {}
    for file in os.listdir("splits"):
        with open(os.path.join("splits", file), 'r') as file:
            lines = file.readlines()
            split_name = file.name.split("/")[1].split(".")[0]
            splits[split_name] = [int(line[:-1]) for line in lines] # remove new line characters
    return splits

from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch

# Simple dataset
class CustomDataset(Dataset):
    def __init__(self, dataX, dataY):
        self.X = dataX
        self.y = dataY

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_dims(self):
        return len(self.y[0])

# Single layer NN for testing embeddings
class SingleLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def load_data(embeddings_dir, fa_sol_dir, fa_atr_dir):
    # Load embeddings and inputs
    embeddings = np.load(embeddings_dir, allow_pickle=True)
    fa_sol_inputs = pd.read_csv(fa_sol_dir)
    fa_atr_inputs = pd.read_csv(fa_atr_dir)

    splits = load_splits()

    # Baseline embeddings
    file_idx = 0 if embeddings.files[0] == 'X' else 1
    X = np.array(embeddings[embeddings.files[file_idx]])
    X_train = X[splits["train"]]
    X_test = X[splits["test"]]
    X_val = X[splits["val"]]

    # embeddings + fa_sol
    X_gfp = np.column_stack((X, np.array(fa_sol_inputs['fa_sol'].astype(np.float32))))
    X_gfp_train = X_gfp[splits["train"]]
    X_gfp_test = X_gfp[splits["test"]]
    X_gfp_val = X_gfp[splits["val"]]

    # Test prediction of Rosetta attraction energy
    y = np.array(fa_atr_inputs['fa_atr'])
    y_train = y[splits["train"]]
    y_test = y[splits["test"]]
    y_val = y[splits["val"]]

    # DataLoaders
    train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=64, shuffle=True)
    train_gfp_loader = DataLoader(CustomDataset(X_gfp_train, y_train), batch_size=64, shuffle=True)

    val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=64, shuffle=True)
    val_gfp_loader = DataLoader(CustomDataset(X_gfp_val, y_val), batch_size=64, shuffle=True)

    data_dim = X.shape[1]

    return {'train_loader': train_loader,
            'train_gfp_loader': train_gfp_loader,
            'val_loader': val_loader,
            'val_gfp_loader': val_gfp_loader,
            'X_test': torch.tensor(X_test),
            'X_gfp_test': torch.tensor(X_gfp_test),
            'y_test': torch.tensor(y_test)}, data_dim

import torch

def train(model, epochs, training_loader, val_loader, loss_fn, optimizer, verbose=False):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        # Train
        for inputs, labels in training_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(np.squeeze(outputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(training_loader)
        if verbose:
            print(f'  Epoch {epoch + 1} train loss: {train_loss}')
        train_losses.append(train_loss)

        # Validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                model.eval()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(np.squeeze(outputs), labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        if verbose:
            print(f'  Epoch {epoch + 1} val loss: {val_loss}')
        val_losses.append(val_loss)
    
    return train_losses, val_losses

def compare(embeddings_dir, fa_sol_dir, fa_atr_dir, model_str, epochs=10, lr=0.01):
    data, input_dim  = load_data(embeddings_dir, fa_sol_dir, fa_atr_dir)

    # Baseline embeddings
    model = SingleLayer(input_dim)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = train(model, epochs,
                                    data['train_loader'],
                                    data['val_loader'],
                                    loss_fn, optimizer)

    # Embeddings + solvation energy
    model_gfp = SingleLayer(input_dim+1) # one additional parameter
    optimizer = torch.optim.Adam(model_gfp.parameters(), lr=lr)
    train_gfp_losses, val_gfp_losses = train(model_gfp, epochs,
                                             data['train_gfp_loader'],
                                             data['val_gfp_loader'],
                                             loss_fn, optimizer)
    test_outputs = model(data['X_test'])
    print(test_outputs.shape)
    test_loss = loss_fn(np.squeeze(test_outputs), data['y_test'])

    test_gfp_outputs = model_gfp(data['X_gfp_test'])
    test_gfp_loss = loss_fn(np.squeeze(test_gfp_outputs), data['y_test'])
    results = {'train_losses': train_losses,
            'train_gfp_losses': train_gfp_losses,
            'val_losses': val_losses,
            'val_gfp_losses': val_gfp_losses,
            'test_loss': test_loss.item(),
            'test_gfp_loss': test_gfp_loss.item()}

    plot_comparison(model_str, results)

    return results


import matplotlib.pyplot as plt
def plot_comparison(model_str, losses):
    runs = {
    f"{model_str} Embeddings": {"train": losses['train_losses'],
                                "val": losses['val_losses'],
                                "test": losses['test_loss']},
    f"{model_str} Embeddings + Solvation Energies": {"train": losses['train_gfp_losses'],
                                                     "val": losses['val_gfp_losses'],
                                                     "test": losses['test_gfp_loss']},
}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (name, hist) in zip(axes, runs.items()):
        ax.set_yscale("log")
        epochs = np.arange(1, len(hist["train"]) + 1)
        ax.plot(epochs, hist["train"], label="Train", color="tab:blue")
        ax.plot(epochs, hist["val"], label="Validation", color="tab:orange")
        ax.scatter(len(hist["train"]), hist["test"],
                   label=f"Test Loss: {hist['test']:.3f}",
                   color="tab:green")

        # Mark the best validation epoch
        best = int(np.argmin(hist["val"]))
        ax.scatter(best + 1, hist["val"][best], color="tab:red", zorder=3,
                label=f"Best val ({hist['val'][best]:.3f}, epoch {best + 1})")

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()


    axes[0].set_ylabel("Loss")
    fig.suptitle("Loss comparison", fontsize=13)
    fig.tight_layout()
    plt.show()