import torch, torchvision
from torch import nn, optim
from torch.autograd import Variable as var 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data

### Some parameters
dropout_p = 0.2
num_hidden_units = 50
num_classes = 10 # MNIST

### CNN class for handwritten digits recognition
class digCNN(nn.Module):
    def __init__(self):
        super(digCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout_conv2 = nn.Dropout2d(dropout_p)
        self.fc1 = nn.Linear(1024, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)  # flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x