# =============================== #
#         Data Parallelism        #
# =============================== #

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================== #
#    Parameters and DataLoaders   #
# =============================== #

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dummy Dataset
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


dummy_loader = DataLoader(
    dataset=RandomDataset(input_size, data_size),
    batch_size=batch_size,
    shuffle=True
)


# Dummy Model
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


# =============================== #
#  Create Model and DataParallel  #
# =============================== #

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

# Run the Model
for data in dummy_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
