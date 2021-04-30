# =============================== #
#         Import Libraries        #
# =============================== #

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Hyper parameters
batch_size = 64
epochs = 5

# =============================== #
#            Load Data            #
# =============================== #

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# =============================== #
#        Create Data Loaders      #
# =============================== #

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False)

for X, y in test_dataloader:
    print('Shape of X [N, C, H, W]:', X.size())
    print('Shape of y:', y.size())
    break

"""Result:
            Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
            Shape of y:  torch.Size([64]) torch.int64
"""

# =============================== #
#         Creating Models         #
# =============================== #

# Get CPU or GPU device for training
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')


# Define model
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


model = NeuralNetwork().to(device)
print(model)

# =============================== #
# Optimizing the Model Parameters #
# =============================== #

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
print(optimizer)

# =============================== #
#          Train the Model        #
# =============================== #

size = len(train_dataloader.dataset)
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        # Compute prediction error
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            loss, current = loss.item(), i * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# =============================== #
#          Test the Model         #
# =============================== #

size = len(test_dataloader.dataset)
model.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        test_loss += loss_fn(outputs, labels).item()
        _, predictions = torch.max(outputs.data, 1)

        correct += (predictions == labels).sum().item()

    test_loss = test_loss / size
    correct = correct / size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# =============================== #
#          Saving Models          #
# =============================== #

torch.save(model.state_dict(), "model.pt")
print("Saved PyTorch Model State to model.pt")

# =============================== #
#         Loading Models          #
# =============================== #

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pt"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    predictions = model(x)
    predicted, actual = classes[predictions[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
