import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========================================= #
#           Get Device for Training         #
# ========================================= #

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')


# ========================================= #
#              Define the Class             #
# ========================================= #

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
'''Result:
            NeuralNetwork(
              (flatten): Flatten(start_dim=1, end_dim=-1)
              (linear_relu_stack): Sequential(
                (0): Linear(in_features=784, out_features=512, bias=True)
                (1): ReLU()
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): ReLU()
                (4): Linear(in_features=512, out_features=10, bias=True)
                (5): ReLU()
              )
            )
'''

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probability = nn.Softmax(dim=1)(logits)
y_pred = pred_probability.argmax(1)
print(f"Predicted class: {y_pred}")

''' Predicted class: tensor([9], device='cuda:0') '''

# ========================================= #
#               Model Layers                #
# ========================================= #

input_image = torch.rand(3, 28, 28)
print(input_image.size())
'''Result:
            torch.Size([3, 28, 28])
'''

# nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
'''Result:
            torch.Size([3, 784])
'''

# nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
'''Result:
            torch.Size([3, 20])
'''

# nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

"""
Before ReLU: tensor([[-0.1126, -0.3230,  0.5091,  0.3814, -0.1690,  0.1766,  0.0877,  0.0550,
         -0.1617, -0.2699,  0.1601, -0.5527, -1.1046, -0.0263, -0.1490, -0.2710,
          0.2683, -0.0678,  0.0049, -0.0662],
        [-0.2466,  0.1945,  0.3531,  0.1308,  0.0011, -0.0651,  0.3204, -0.2037,
         -0.3126, -0.0906,  0.0174, -0.5440, -1.0114,  0.3616, -0.2000, -0.0712,
          0.2265, -0.3949, -0.0954, -0.0048],
        [-0.1062,  0.0397,  0.1318,  0.2476, -0.1244, -0.2751,  0.0455, -0.2235,
         -0.4011,  0.0195,  0.1683, -0.9019, -0.7498,  0.5108, -0.3084, -0.1637,
          0.1855, -0.5018,  0.0173, -0.3605]], grad_fn=<AddmmBackward>)


After ReLU: tensor([[0.0000, 0.0000, 0.5091, 0.3814, 0.0000, 0.1766, 0.0877, 0.0550, 0.0000,
         0.0000, 0.1601, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2683, 0.0000,
         0.0049, 0.0000],
        [0.0000, 0.1945, 0.3531, 0.1308, 0.0011, 0.0000, 0.3204, 0.0000, 0.0000,
         0.0000, 0.0174, 0.0000, 0.0000, 0.3616, 0.0000, 0.0000, 0.2265, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0397, 0.1318, 0.2476, 0.0000, 0.0000, 0.0455, 0.0000, 0.0000,
         0.0195, 0.1683, 0.0000, 0.0000, 0.5108, 0.0000, 0.0000, 0.1855, 0.0000,
         0.0173, 0.0000]], grad_fn=<ReluBackward0>)
"""

# nn.Sequential
""" nn.Sequential is an ordered container of modules. The data is passed through 
all the modules in the same order as defined. You can use sequential containers 
to put together a quick network like seq_modules """

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# ========================================= #
#                Model Parameters           #
# ========================================= #

''' Iterating over each parameter, and print its size and a preview of its values '''
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
