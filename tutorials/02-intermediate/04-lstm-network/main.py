# ================================================================ #
#                       LSTM Neural Networks                       #
# ================================================================ #
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms

# Hyper parameters
import tqdm

batch_size = 100
num_epochs = 10
learning_rate = 0.1

input_dim = 28
hidden_dim = 100
sequence_dim = 28
layer_dim = 1
output_dim = 10

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================ #
#                        Data Loading Process                      #
# ================================================================ #

# Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ================================================================ #
#                           Train and Test                         #
# ================================================================ #

# Train the model
iter = 0
print('TRAINING STARTED.\n')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, sequence_dim, input_dim).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # Calculate Loss
            print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.2f}')

# Test the model
model.eval()
print('\nCALCULATING ACCURACY...\n')
with torch.no_grad():
    correct = 0
    total = 0
    progress = tqdm.tqdm(test_loader, total=len(test_loader))
    # Iterate through test dataset
    for images, labels in progress:
        images = images.view(-1, sequence_dim, input_dim).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Print Accuracy
    print(f'Accuracy: {accuracy}')
