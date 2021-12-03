# ================================================================ #
#                         Define the Network                       #
# ================================================================ #

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

'''Result:
    Net(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
'''

# The learnable parameters of a model are returned by `net.parameters()`
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32. To use this net on the
# MNIST dataset, please resize the images from the dataset to 32x32

input = torch.randn(1, 1, 32, 32)  # dummy image
out = net(input)
print(out)

'''Result:
        tensor([[-0.0381, -0.1305,  0.0026, -0.1244,  0.0164, -0.0031, -0.0207, -0.0518,
              0.0536, -0.0641]], grad_fn=<AddmmBackward0>)
'''

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# ================================================================ #
#                            Loss Function                         #
# ================================================================ #

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
'''Result:
        tensor(0.8491, grad_fn=<MseLossBackward0>)
'''

'''If you follow loss in the backward direction, using its .grad_fn attribute, you will see a graph of computations that 
looks like this:
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
'''

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

'''Result:
        <MseLossBackward0 object at 0x7f0e70779f60>
        <AddmmBackward0 object at 0x7f0e70779080>
        <AccumulateGrad object at 0x7f0e70779080>
'''

# ================================================================ #
#                          Backpropagation                         #
# ================================================================ #

# Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.

net.zero_grad()  # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

'''Result:
        conv1.bias.grad before backward
        tensor([0., 0., 0., 0., 0., 0.])
        conv1.bias.grad after backward
        tensor([-0.0103, -0.0144, -0.0100,  0.0006, -0.0013, -0.0019])
'''

# ================================================================ #
#                         Update the weights                       #
# ================================================================ #

# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD): `weight = weight - learning_rate * gradient`

# We can implement this using simple Python code:
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

"""
However, as you use neural networks, you want to use various different update rules such as SGD, Nesterov-SGD, 
Adam, RMSProp, etc. To enable this, we built a small package: torch.optim that implements all these methods. Using it 
is very simple: 
"""
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update
