![logo](logo/pytorch_logo.svg)

# PyTorch Tutorials

A comprehensive collection of PyTorch tutorials ranging from basics to advanced topics. This repository is designed to help both beginners and advanced users learn PyTorch through practical, hands-on examples.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Tutorial Contents](#-tutorial-contents)
  - [Basics](#basics)
  - [Beginner](#beginner)
  - [Intermediate](#intermediate)
  - [Advanced](#advanced)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Sample Output](#-sample-output)
- [FAQ & Troubleshooting](#-faq--troubleshooting)
- [Roadmap](#-roadmap)
- [References](#-references)
- [Contributing](#-contributing)

---

## ✨ Features

- **Progressive Learning Path**: Tutorials organized from basics to advanced topics
- **Dual Format**: Each tutorial available as both Python script and Jupyter notebook
- **Practical Examples**: Real-world use cases with popular datasets (MNIST, CIFAR-10, Fashion-MNIST)
- **Well-Commented Code**: Detailed explanations and comments throughout
- **Auto-Download Datasets**: Automatic dataset downloading where applicable
- **Modern PyTorch Practices**: Uses current PyTorch APIs and best practices

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.7 or higher
- **PyTorch**: 1.7 or higher (1.10+ recommended)
- **Operating Systems**: Linux, macOS, or Windows

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yakhyo/pytorch-tutorials.git
   cd pytorch-tutorials
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv pytorch-env
   source pytorch-env/bin/activate  # On Windows: pytorch-env\Scripts\activate
   
   # Or using conda
   conda create -n pytorch-env python=3.9
   conda activate pytorch-env
   ```

3. **Install PyTorch**:
   
   Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation commands specific to your system. For example:
   
   ```bash
   # CPU only
   pip install torch torchvision
   
   # With CUDA support (for GPU)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install additional dependencies**:
   ```bash
   pip install matplotlib numpy jupyter
   ```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

## 🚀 Quick Start

### Running Python Scripts

Navigate to any tutorial directory and run the Python script directly:

```bash
# Example: Run the QuickStart tutorial
cd tutorials/01-basics/01-quickstart
python main.py

# Example: Run CNN on CIFAR-10
cd tutorials/03-intermediate/01-convolutional-nn
python main.py
```

### Running Jupyter Notebooks

Launch Jupyter and navigate to any tutorial:

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Then navigate to the desired tutorial, for example:
- `tutorials/01-basics/01-quickstart/main.ipynb`

### Your First Model in 5 Minutes

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Create data loader
train_dataloader = DataLoader(training_data, batch_size=64)

# Define a simple neural network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# Train and evaluate (see full tutorials for complete code)
```

---

## 📚 Tutorial Contents

### Basics

Start here if you're new to PyTorch! These tutorials cover fundamental concepts.

- **QuickStart** [[Script](tutorials/01-basics/01-quickstart/main.py) | [Notebook](tutorials/01-basics/01-quickstart/main.ipynb)]  
  A comprehensive overview of the basics: loading data, building models, training, and inference.

- **Tensors** [[Script](tutorials/01-basics/02-tensors/main.py) | [Notebook](tutorials/01-basics/02-tensors/main.ipynb)]  
  Operations on tensors, working with NumPy arrays, and conversions between them.

- **Datasets and DataLoaders** [[Script](tutorials/01-basics/03-dataset-dataloaders/main.py) | [Notebook](tutorials/01-basics/03-dataset-dataloaders/main.ipynb)]  
  Creating custom datasets and efficient data loading pipelines.

- **Transforms** [[Script](tutorials/01-basics/04-transforms/main.py) | [Notebook](tutorials/01-basics/04-transforms/main.ipynb)]  
  Data augmentation and transformation techniques using torchvision.

- **Build Neural Network** [[Script](tutorials/01-basics/05-neural-network/main.py) | [Notebook](tutorials/01-basics/05-neural-network/main.ipynb)]  
  Building and understanding neural network architectures from scratch.

- **Optimizers, Model Save and Load** [[Script](tutorials/01-basics/06-optim-save-load/main.py) | [Notebook](tutorials/01-basics/06-optim-save-load/main.ipynb)]  
  Working with optimizers, saving trained models, and loading for inference.

### Beginner

Build on the basics with these intermediate concepts.

- **Autograd and Freeze Weights** [[Script](tutorials/02-beginner/01-autograd/main.py) | [Notebook](tutorials/02-beginner/01-autograd/main.ipynb)]  
  Understanding automatic differentiation and gradient computation.

- **Neural Networks Deep Dive** [[Script](tutorials/02-beginner/02-neural-networks/main.py) | [Notebook](tutorials/02-beginner/02-neural-networks/main.ipynb)]  
  In-depth exploration of neural network components and operations.

### Intermediate

Advanced architectures and techniques for real-world applications.

- **Convolutional Neural Network** [[Script](tutorials/03-intermediate/01-convolutional-nn/main.py)]  
  Image classification on CIFAR-10 using CNNs with convolutional layers.

- **Residual Neural Network** [[Script](tutorials/03-intermediate/02-deep-residual-nn/main.py)]  
  Deep networks with skip connections (ResNet) for improved training.

- **Recurrent Neural Network** [[Script](tutorials/03-intermediate/03-recurrent-nn/main.py)]  
  Sequence processing and image classification with RNN architectures.

- **Long Short Term Memory** [[Script](tutorials/03-intermediate/04-lstm-network/main.py)]  
  MNIST digit classification using LSTM networks to handle sequences.

- **Variational Auto Encoder** [[Script](tutorials/03-intermediate/05-var-auto-encode/main.py)]  
  Generative modeling with VAE for reconstructing MNIST samples.

### Advanced

State-of-the-art techniques and production deployment.

- **Adversarial Generative Network** [[External: DCGAN Repository](https://github.com/yakhyo/DCGAN-pt)]  
  Deep Convolutional Generative Adversarial Network for image generation.

- **Fast Neural Style Transfer** [[External: Style Transfer Repository](https://github.com/yakhyo/Fast-Neural-Style-Transfer)]  
  Real-time artistic style transfer to images.

- **Super Resolution with ONNX** [[Script](tutorials/04-advanced/03-super-resolution-onnx/main.py)]  
  Model inference using ONNX Runtime for production deployment.

---

## 📁 Project Structure

```
pytorch-tutorials/
├── logo/
│   └── pytorch_logo.svg          # Project logo
├── tutorials/
│   ├── 01-basics/                # Fundamental PyTorch concepts
│   │   ├── 01-quickstart/
│   │   ├── 02-tensors/
│   │   ├── 03-dataset-dataloaders/
│   │   ├── 04-transforms/
│   │   ├── 05-neural-network/
│   │   └── 06-optim-save-load/
│   ├── 02-beginner/              # Beginner-level tutorials
│   │   ├── 01-autograd/
│   │   └── 02-neural-networks/
│   ├── 03-intermediate/          # Intermediate architectures
│   │   ├── 01-convolutional-nn/
│   │   ├── 02-deep-residual-nn/
│   │   ├── 03-recurrent-nn/
│   │   ├── 04-lstm-network/
│   │   └── 05-var-auto-encode/
│   └── 04-advanced/              # Advanced topics
│       ├── 01-adversarial-network/
│       ├── 02-neural-style-transfer/
│       └── 03-super-resolution-onnx/
├── LICENSE                       # MIT License
└── README.md                     # This file
```

Each tutorial directory contains:
- `main.py` - Python script version
- `main.ipynb` - Jupyter notebook version (where available)

---

## 📊 Datasets

### Automatic Download

Most tutorials use popular datasets that download automatically:

- **Fashion-MNIST**: 60,000 training + 10,000 test images (28x28 grayscale)
- **MNIST**: Handwritten digits dataset
- **CIFAR-10**: 60,000 color images in 10 classes (32x32)

These datasets are automatically downloaded to the `data/` directory when you run the tutorials for the first time.

### Custom Datasets

To use your own datasets:

1. Organize your data in a folder structure:
   ```
   your_dataset/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   └── test/
       ├── class1/
       ├── class2/
       └── ...
   ```

2. Use PyTorch's `ImageFolder` or create a custom `Dataset` class:
   ```python
   from torchvision.datasets import ImageFolder
   from torchvision import transforms
   
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
   ])
   
   dataset = ImageFolder(root='path/to/your_dataset/train', transform=transform)
   ```

For more details, see the [Datasets and DataLoaders tutorial](tutorials/01-basics/03-dataset-dataloaders/).

---

## 🖼️ Sample Output

**Example: Fashion-MNIST Classification**

After training a simple neural network on Fashion-MNIST, you can expect outputs like:

```
Epoch [1/5], Loss: 0.5123
Epoch [2/5], Loss: 0.3856
Epoch [3/5], Loss: 0.3421
Epoch [4/5], Loss: 0.3187
Epoch [5/5], Loss: 0.3012

Test Accuracy: 88.5%

Predicted: Ankle boot, Actual: Ankle boot ✓
Predicted: T-shirt/top, Actual: T-shirt/top ✓
Predicted: Pullover, Actual: Pullover ✓
```

The tutorials include visualization code to display sample predictions and training curves.

---

## ❓ FAQ & Troubleshooting

### Common Issues

**Q: ImportError: No module named 'torch'**

A: Make sure PyTorch is installed in your active Python environment:
```bash
pip install torch torchvision
```

**Q: CUDA out of memory error**

A: Reduce the batch size in the training script. For example, change:
```python
batch_size = 64  # Change to 32 or 16
```

**Q: Downloads are slow or timing out**

A: The first run downloads datasets which may take time. If it fails:
- Check your internet connection
- Try running again (downloads resume where they left off)
- Manually download datasets and place in the `data/` folder

**Q: "RuntimeError: Expected all tensors to be on the same device"**

A: Ensure your model and data are on the same device:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
data = data.to(device)
```

**Q: Jupyter notebook kernel crashes**

A: This usually happens with large models or datasets. Try:
- Restart the kernel and clear all outputs
- Reduce batch size or model size
- Close other applications to free memory

**Q: Permission denied when saving models**

A: Make sure you have write permissions in the directory:
```bash
chmod +w ./  # On Linux/macOS
```

### Tips for Beginners

- **Start with Basics**: Complete the `01-basics` tutorials in order before moving to advanced topics.
- **Read the Comments**: Code is heavily commented to explain each step.
- **Experiment**: Modify hyperparameters (learning rate, batch size, epochs) to see their effects.
- **Use GPU**: If available, training on GPU is much faster. Check with `torch.cuda.is_available()`.
- **Monitor Training**: Watch loss values - they should generally decrease over epochs.

### Getting Help

- **PyTorch Forums**: [discuss.pytorch.org](https://discuss.pytorch.org/)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
- **Stack Overflow**: Tag your questions with `pytorch`
- **GitHub Issues**: Open an issue in this repository for tutorial-specific problems

---

## 🗺️ Roadmap

We're continuously improving and expanding these tutorials. Upcoming additions:

### Coming Soon
- [ ] **Transformers and Attention Mechanisms**: Self-attention, multi-head attention, and Transformer architecture
- [ ] **Vision Transformers (ViT)**: Modern computer vision with transformers
- [ ] **Natural Language Processing**: Text classification, sentiment analysis, and language models
- [ ] **Transfer Learning**: Fine-tuning pre-trained models (ResNet, BERT, GPT)
- [ ] **Object Detection**: YOLO, Faster R-CNN implementations
- [ ] **Distributed Training**: Multi-GPU and multi-node training
- [ ] **Model Optimization**: Quantization, pruning, and knowledge distillation
- [ ] **MLOps Integration**: Model deployment, monitoring, and versioning

### Enhancement Plans
- [ ] Add more visualization examples and outputs
- [ ] Create video walkthroughs for complex topics
- [ ] Add performance benchmarking scripts
- [ ] Include pre-trained model downloads
- [ ] Add Docker containerization for easy setup

**Suggestions?** We'd love to hear what tutorials you'd like to see next! Open an issue with your ideas.

---

## 📖 References

### Official Documentation
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)

### Learning Resources
- [PyTorch Examples Repository](https://github.com/pytorch/examples)
- [Deep Learning Wizard](https://www.deeplearningwizard.com/deep_learning/intro/)
- [Dive into Deep Learning](https://d2l.ai/) - Interactive deep learning book
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS230: Deep Learning](https://cs230.stanford.edu/)

### Research Papers
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)
- [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661)

### Community
- [PyTorch Discuss Forum](https://discuss.pytorch.org/)
- [PyTorch Slack Community](https://pytorch.slack.com/)
- [r/PyTorch on Reddit](https://www.reddit.com/r/pytorch/)

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new tutorials, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-tutorial`
3. **Make your changes**: Follow the existing code style and structure
4. **Test your changes**: Ensure code runs without errors
5. **Commit your changes**: `git commit -m "Add new tutorial on [topic]"`
6. **Push to your fork**: `git push origin feature/new-tutorial`
7. **Open a Pull Request**: Describe your changes clearly

### Contribution Guidelines

- Keep tutorials clear, well-commented, and beginner-friendly
- Include both `.py` script and `.ipynb` notebook versions
- Follow PEP 8 style guide for Python code
- Test on multiple Python/PyTorch versions when possible
- Update this README if adding new tutorials or sections

### Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for everyone. Please be respectful and constructive in all interactions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

Special thanks to:
- The PyTorch team for the excellent framework and documentation
- All contributors who have helped improve these tutorials
- The deep learning community for inspiration and knowledge sharing

---

<div align="center">

**Happy Learning! 🚀**

*Star ⭐ this repository if you find it helpful!*

*Questions or feedback? Open an issue or start a discussion.*

**[⬆ back to top](#pytorch-tutorials)**

</div>
