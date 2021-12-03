# ================================================================ #
#                           Torch Autograd                         #
# ================================================================ #

import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# We create `Q` tensor using `a` and `b` tensors

Q = 3 * a ** 3 - b ** 2

# Let's assume `a` and `b` are parameters of the model and `Q` is a error function
# During the training we calculate the gradients w.r.t parameters `a`, `b`

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# When .backward() means taking a derivative of Q w.r.t `a`, `b`
# Gradients are now deposited in `a.grad` and `b.grad`

# Let's check the derivatives, f(a) = 3*a**3 -> f`(a) = 9*a**2, f(b) = b**2 -> f`(b) = 2*b
print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)
'''Result:
        tensor([True, True])
        tensor([True, True])
'''

# In above example we see the derivatives because the tensor `a` and `b` have `requires_grad=True` parameter. For this
# reason gradient was calculated if we do not set `requires_grad=True` then there will not be any gradient calculations for these variables.

# In the example below x and y have `requires_grad=False` by default

# The output tensor of an operation will require gradients even if only a single input tensor has ``requires_grad=True`
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y  # False + False = False
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z  # False + True = True
print(f"Does `b` require gradients?: {b.requires_grad}")

'''Result:
        Does `a` require gradients? : False
        Does `b` require gradients?: True
'''

# ================================================================ #
#                 Finetunig, Freeze the model weights              #
# ================================================================ #

# In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels.

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False


# Let's say we want to finetune the model on a new dataset with 10 labels.
# In resnet, the classifier is the last linear layer ``model.fc``.
# We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier.

model.fc = nn.Linear(512, 10)


# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
