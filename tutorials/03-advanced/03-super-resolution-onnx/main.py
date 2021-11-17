import io
import numpy as np

import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init


# ================================================================ #
#                         Building the Model                       #
# ================================================================ #

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


# Creating an instance from SuperResolutionNet
net = SuperResolutionNet(upscale_factor=3)

# ================================================================ #
#                  Downloading Pretrained Weights                  #
# ================================================================ #

model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

# Initialize model with the pretrained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(model_zoo.load_url(model_url, map_location=device))
net.eval()  # Changing to eval mode to save it onnx format

# onnx input shape: x.shape : (batch_size=1, channel=1, H, W)
# The model expects the Y component of the YCbCr of an image as an input so it has one channel
x = torch.randn(1, 1, 224, 224, requires_grad=True)
onnx_model = net(x)
"""
# Export the onnx model
torch.onnx.export(onnx_model,                                # model being run
                  x,                                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",                   # where to save the model
                  export_params=True,                        # store the trained parameter weights inside the model file
                  opset_version=10,                          # the ONNX version to export the model to
                  do_constant_folding=True,                  # whether to execute constant folding for optimization
                  input_names=['input'],                     # the model's input names
                  output_names=['output'],                   # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
"""
# ================================================================ #
#                         Loading ONNX model                       #
# ================================================================ #

import onnx
import onnxruntime

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# ================================================================ #
#           Reading Original Image and Feed it to Model            #
# ================================================================ #

from PIL import Image
import torchvision.transforms as transforms

img = Image.open("../../../cat_224x224.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

# The model expects the Y component of the YCbCr of an image as an input

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
output = Image.merge(
    "YCbCr",
    [img_out_y, img_cb.resize(img_out_y.size, Image.BICUBIC), img_cr.resize(img_out_y.size, Image.BICUBIC), ]
).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
output.save("../../../cat_superres_with_ort.jpg")
