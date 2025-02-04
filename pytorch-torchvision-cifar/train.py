import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

from azureml.core import Run

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point")
parser.add_argument("--learning-rate", type=float, dest="lr", default=0.001, help="learning rate")
parser.add_argument("--weight-decay", type=float, dest="weight_decay", default=0.0, help="weight decay (L2 regularization)")
parser.add_argument("--epochs", type=int, dest="epochs", default=5, help="Number of epochs to train the model for")
args = parser.parse_args()

data_folder = args.data_folder
epochs = args.epochs
lr = args.lr
wd = args.weight_decay
print(f"Data folder: {data_folder}\n")

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.CIFAR10(
    root=data_folder,
    train=True,
    download=True,
    transform=transform
)
testset = datasets.CIFAR10(
    root=data_folder,
    train=False,
    download=True,
    transform=transform
)

trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    drop_last=True
)
testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    drop_last=True
)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=wd
)

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

# Evaluate the model on testset
model.eval()
ypred, ytrue = [], []
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        ypred.extend(predicted.numpy())
        ytrue.extend(labels.numpy())

# Calculate accuracy
acc = accuracy_score(ytrue, ypred)
print(f"Accuracy is: {acc}")

# Log metrics to AML
run.log('epochs', epochs)
run.log('accuracy', acc)

# Save the trained model
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), "outputs/pytorch-torchvision-cifar-model.pth")