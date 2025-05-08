import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 6
input_size = 7  # Should match the number of features in the dataset
hidden_size = 10
num_classes = 1
learning_rate = 0.0001
epochs = 500


class BatteryDataSet(Dataset):
    def __init__(self):
        # Data loading and preprocessing
        dataset_raw = pd.read_csv(os.getcwd() + '/Datasets/HNEI_Processed/Final Database.csv')
        dataset_raw.drop('Unnamed: 0', axis=1, inplace=True)

        # Feature scaling
        data = dataset_raw.values[:, :-1]
        trans = MinMaxScaler()
        data = trans.fit_transform(data)
        dataset = pd.DataFrame(data)
        dataset_scaled = dataset.join(dataset_raw['RUL'])
        scaled_df_np = dataset_scaled.to_numpy(dtype=np.float32)

        # Corrected feature selection (columns 2 to -2)
        self.x = torch.from_numpy(scaled_df_np[:, 2:-1])
        self.y = torch.from_numpy(scaled_df_np[:, [-1]])
        self.n_samples = scaled_df_np.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def create_loaders(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))

    if shuffle:
        np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def train_loop(train_loader, model, loss_fn, optimizer):
    model.train()
    for batch, (features, rul) in enumerate(train_loader):
        # Move data to GPU
        features = features.to(device)
        rul = rul.to(device)

        # Forward pass
        outputs = model(features)
        loss = loss_fn(outputs, rul)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(test_loader, model, loss_fn):
    model.eval()
    test_loss = 0
    diff_list = []
    targets = []
    predictions = []

    with torch.no_grad():
        for features, rul in test_loader:
            # Move data to GPU
            features = features.to(device)
            rul = rul.to(device)

            preds = model(features)
            test_loss += loss_fn(preds, rul).item()

            # Move data back to CPU for calculations
            cpu_rul = rul.cpu().numpy()
            cpu_preds = preds.cpu().numpy()

            diff = np.abs(cpu_rul - cpu_preds) / cpu_rul
            diff_list.extend(diff.flatten())
            targets.extend(cpu_rul.flatten())
            predictions.extend(cpu_preds.flatten())

    avg_diff = np.mean(diff_list) * 100
    avg_loss = test_loss / len(test_loader)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'Prediction vs Actual (Avg Difference: {avg_diff:.2f}%)')
    plt.show()

    print(f"Test Results:\nAvg Difference: {avg_diff:.2f}%\nAvg Loss: {avg_loss:.4f}\n")


if __name__ == "__main__":
    # Initialize dataset and loaders
    dataset = BatteryDataSet()
    train_loader, test_loader = create_loaders(dataset, batch_size)

    # Initialize model and move to GPU
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_diff = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)

    # Save model
    torch.save(model.state_dict(), 'battery_rul_predictor.pth')
    print("Training complete! Model saved.")