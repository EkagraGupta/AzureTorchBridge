import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

from azureml.core import Run

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Parse command-line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder", help="data doler mounting point")
parser.add_argument("--learning-rate", type=float, dest="lr", default=0.001, help="learning rate")
parser.add_argument("--weight decay", type=float, dest="weight_decay", default=0.0, help="weight decay (L2 regularization)")
parser.add_argument("--epochs", type=int, dest="epochs", default=5, help="Number of epochs to train the model for")
args = parser.parse_args()

# Extract arguments
data_folder = args.data_folder
epochs = args.epochs
lr = args.lr
wd = args.weight_decay
print(f"Data folder: {data_folder}")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST data
trainset = datasets.MNIST(root=data_folder,
                          train=True,
                          download=True,
                          transform=transform)
testset = datasets.MNIST(root=data_folder,
                         train=False,
                         download=True,
                         transform=transform)

# Create data loaders
trainloader = DataLoader(trainset,
                            batch_size=64,
                            shuffle=True)
testloader = DataLoader(testset,
                        batch_size=64,
                        shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=wd)

# Get the Azure ML run context for logging
run = Run.get_context()

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")


# Evaluate the model on the test set
model.eval()
ypred, ytrue = [], []
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        ypred.extend(predicted.numpy())
        ytrue.extend(labels.numpy())

# Calculate the accuracy
acc = accuracy_score(ytrue, ypred)
print(f"Accuracy is: {acc}")

# Log metrics to Azure ML
run.log("epochs", epochs)
run.log("accuracy", acc)

# Save the trained model
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/pytorch-torchvision-mnist-model.pth")
