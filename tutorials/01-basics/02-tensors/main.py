import torch
import numpy as np

# =============================== #
#     Initializing a Tensor       #
# =============================== #

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f'Ones Tensor: \n {x_ones}')

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f'Random Tensor: \n {x_rand}')

'''Result:
            Ones Tensor: 
             tensor([[1, 1],
                    [1, 1]])
            Random Tensor: 
             tensor([[0.8154, 0.5755],
                    [0.3316, 0.8800]])
'''

# With random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor: \n {rand_tensor}')
print(f'Ones Tensor: \n {ones_tensor}')
print(f'Zeros Tensor: \n {zeros_tensor}')

'''Result:
            Random Tensor: 
             tensor([[0.8694, 0.3774, 0.8403],
                    [0.5461, 0.7127, 0.1988]])
            Ones Tensor: 
             tensor([[1., 1., 1.],
                    [1., 1., 1.]])
            Zeros Tensor: 
             tensor([[0., 0., 0.],
                    [0., 0., 0.]])
'''

# =============================== #
#     Attributes of a Tensor      #
# =============================== #

''' Tensor attributes describe their shape, datatype, and the device on which they are stored '''

tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')

'''Result:
            Shape of tensor: torch.Size([3, 4])
            Datatype of tensor: torch.float32
            Device tensor is stored on: cpu
'''

# =============================== #
#       Operations on Tensor      #
# =============================== #


# Moving a tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Standard numpy-like indexing and slicing
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=torch.float)
print(f'First row: {tensor[0]}')
print(f'First column: {tensor[:, 0]}')
print(f'Last column: {tensor[..., -1]}')
tensor[:, 1] = 0
print(tensor)

'''Result:
            First row: tensor([1., 2., 3., 4.])
            First column: tensor([ 1.,  5.,  9., 13.])
            Last column: tensor([ 4.,  8., 12., 16.])
            tensor([[ 1.,  0,  3.,  4.],
                    [ 5.,  0,  7.,  8.],
                    [ 9.,  0., 11., 12.],
                    [13.,  0., 15., 16.]])
'''

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor, tensor], dim=1)
print(t1)

'''Result:
tensor([[ 1.,  0.,  3.,  4.,  1.,  0.,  3.,  4.,  1.,  0.,  3.,  4.,  1.,  0.,  3.,  4.],
        [ 5.,  0.,  7.,  8.,  5.,  0.,  7.,  8.,  5.,  0.,  7.,  8.,  5.,  0.,  7.,  8.],
        [ 9.,  0., 11., 12.,  9.,  0., 11., 12.,  9.,  0., 11., 12.,  9.,  0., 11., 12.],
        [13.,  0., 15., 16., 13.,  0., 15., 16., 13.,  0., 15., 16., 13.,  0., 15., 16.]])
'''

# Arithmetic operations
''' This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value '''
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

print(f'y1: {y1}')
print(f'y2: {y2}')
print(f'y3: {y3}')

'''Result:
            tensor([[ 26.,  58.,  90., 122.],
                    [ 58., 138., 218., 298.],
                    [ 90., 218., 346., 474.],
                    [122., 298., 474., 650.]])
'''

''' This computes the element-wise product. z1, z2, z3 will have the same value '''
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f'z1: {z1}')
print(f'z2: {z2}')
print(f'z3: {z3}')

'''Result:
            tensor([[  1.,   0.,   9.,  16.],
                    [ 25.,   0.,  49.,  64.],
                    [ 81.,   0., 121., 144.],
                    [169.,   0., 225., 256.]])
'''

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
'''Result:
            104.0 <class 'float'>
'''

# In-place operations
''' Operations that store the result into the operand 
    are called in-place. They are denoted by a _ suffix. 
    For example: x.copy_(y), x.t_(), will change x.
'''

tensor = torch.ones_like(tensor)
print(tensor, '\n')
tensor.add_(5)
print(tensor)

'''Result:
            tensor([[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]]) 
            
            tensor([[6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.],
                    [6., 6., 6., 6.]])
'''

# =============================== #
#        Bridge with NumPy        #
# =============================== #

# Tensor to NumPy
'''Tensors on the CPU and NumPy arrays can 
share their underlying memory locations, and 
changing one will change the other.
'''

t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

'''Result:
            t: tensor([1., 1., 1., 1., 1.])
            n: [1. 1. 1. 1. 1.]
'''

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

'''Result:
            t: tensor([2., 2., 2., 2., 2.])
            n: [2. 2. 2. 2. 2.]
'''

# =============================== #
#     NumPy array to Tensor       #
# =============================== #

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)

print(f't: {t}')
print(f'n: {n}')

'''Result:
            t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
            n: [2. 2. 2. 2. 2.]
'''