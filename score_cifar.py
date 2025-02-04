import json
import numpy as np
import os
import torch
from torch import nn

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
    

def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "pytorch-torchvision-cifar-model.pth"
    )
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

def run(raw_data):
    model.eval()
    data = np.array(json.loads(raw_data)['data'])
    data = data.reshape(-1, 3, 32, 32)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(data_tensor)
        _, yhat = torch.max(outputs, 1)
    
    return yhat.tolist()