# ================================================================ #
#                   Recurrent Neural Networks                      #
# ================================================================ #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.01
sequence_dim = 28
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

# ================================================================ #
#                             Load Data                            #
# ================================================================ #

# Dataset

train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

# Data Loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class RNN(nn.Module):
    def __init__(self, input_dim, hiddin_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        # Hidden dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Recurrent layer
        self.rnn = nn.RNN(input_dim, hiddin_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Output layer
        self.fc = nn.Linear(hiddin_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Forward propagate RNN
        out, hn = self.rnn(x, h0.detach())

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


model = RNN(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Instantiate Optimizer and Loss

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ================================================================ #
#                       Training the Model                         #
# ================================================================ #


iterations = 0
for epoch in range(num_epochs):
    # Training
    for i, (images, labels) in enumerate(train_loader):
        model.train()

        images = images.view(-1, sequence_dim, input_dim).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Loss calculate
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        # Backward pass
        loss.backward()
        optimizer.step()
        iterations += 1

        if iterations % 500 == 0:
            model.eval()
            correct = 0
            total = 0
            # Testing
            for images, labels in test_loader:
                images = images.view(-1, sequence_dim, input_dim).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # Print Loss, Accuracy
            print('Iteration: {}\t Loss: {}\t Accuracy: {}\t'.format(iterations, loss.item(), accuracy))