
import json
import numpy as np
import os
import torch
from torch import nn

# Define the same CNN model architecture used for training
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
    

# Initialize the model and load the trained weights
def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "pytorch-torchvision-mnist-model.pth")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Ren inference on input data
def run(raw_data):
    model.eval()
    data = np.array(json.loads(raw_data)['data'])
    data = data.reshape(-1, 1, 28, 28)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(data_tensor)
        _, yhat = torch.max(outputs, 1)
    
    return yhat.tolist()
