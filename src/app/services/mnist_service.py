import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
import os


class PredictionRequest(BaseModel):
    image: List[float]  # Expecting 784 pixel values (28x28)


def load_data(
    train_path: str, test_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MNIST train and test datasets from CSV."""
    try:
        # Load train dataset
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
        y_train = train_df["label"].values.astype(np.int64)

        # Load test dataset
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop("label", axis=1).values.astype(np.float32) / 255.0
        y_test = test_df["label"].values.astype(np.int64)

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


def create_data_loaders(
    X_train, y_train, X_test, y_test, batch_size: int = 64
) -> tuple:
    """Create DataLoader objects for training and testing."""
    # Convert to PyTorch tensors and reshape for CNN
    X_train = torch.FloatTensor(X_train).reshape(-1, 1, 28, 28)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test).reshape(-1, 1, 28, 28)
    y_test = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


class CNN(nn.Module):
    """Define CNN architecture for MNIST."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    epochs: int = 5,
):
    """Train the CNN model."""
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data, target
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model on test data and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def predict(request: PredictionRequest):
    """Predict digit from input image."""
    try:
        # Load model
        model = CNN()
        model.load_state_dict(torch.load("mnist_cnn_model.pth"))
        model.eval()

        # Process input
        image = np.array(request.image, dtype=np.float32)
        if len(image) != 784:
            raise HTTPException(
                status_code=400, detail="Input must be 784 pixel values"
            )
        image = torch.FloatTensor(image).reshape(1, 1, 28, 28)

        # Predict
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main_fun():
    """Main function to train and save the model."""
    # global device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess train and test data
    print("-------------Loading Dataset")
    X_train, y_train, X_test, y_test = load_data(
        "../data/dataset/archive/mnist_train.csv",
        "../data/dataset/archive/mnist_test.csv",
    )

    print("--------------Creating Data Loaders")
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)

    print("---------------Model Initialization")
    # Initialize model, loss, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("-------------Model Training")
    # Train model
    train_model(model, train_loader, criterion, optimizer)

    print("-----------Model Eveluating")
    # Evaluate model
    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), "mnist_cnn_model.pth")
    return accuracy
