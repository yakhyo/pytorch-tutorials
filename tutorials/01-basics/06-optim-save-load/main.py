# ================================================================ #
#                           Optimization                           #
# ================================================================ #

# Prerequisite Code

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import torch.onnx as onnx
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
loss_fn = nn.CrossEntropyLoss()


# ================================================================ #
#                             Optimizer                            #
# ================================================================ #


# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ================================================================ #
#                          Full Implementation                     #
# ================================================================ #

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# ================================================================ #
#                               Training                           #
# ================================================================ #

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# ================================================================ #
#                       Save and Load the Model                    #
# ================================================================ #

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')  # save the model

''' To load model weights, you need to create an instance 
of the same model first, and then load the parameters using 
load_state_dict() method. '''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.vgg16()  # load the model structure not weights
# To laod the model to cpu or gpu: map_location=device
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

''' Be sure to call model.eval() method before inferencing to set the dropout 
and batch normalization layers to evaluation mode. Failing to do this will 
yield inconsistent inference results.
'''

# ================================================================ #
#                   Deep Save and Load the Models                  #
# ================================================================ #

# save
torch.save(model, 'model.pth')

# load
model = torch.load('model.pth')

# ================================================================ #
#                        Export Model to ONNX                      #
# ================================================================ #

input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')
