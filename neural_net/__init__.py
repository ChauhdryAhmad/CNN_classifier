"""
Neural Network from Scratch - Playing Cards Classifier

A pure Python/Numpy implementation of convolutional neural networks for classifying playing cards.
"""

from .network import Sequential
from .layers import Conv2D, MaxPool2D, Flatten, Dense
from .activations import ReLU, Softmax
from .losses import CrossEntropyLoss
from .optimizers import SGD, Adam
from .utils import one_hot_encode, accuracy, train_test_split, normalize_images

__version__ = "0.1.0"
__all__ = [
    "Sequential",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "Dense",
    "ReLU",
    "Softmax",
    "CrossEntropyLoss",
    "SGD",
    "Adam",
    "one_hot_encode",
    "accuracy",
    "train_test_split",
    "normalize_images",
]
