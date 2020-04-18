import pickle

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine, fetch_lfw_people
import torch
import torchvision
import torchvision.transforms as transforms


def _preprocess(X):
    # scaling: Divide every sample by the root sum of squared elements
    # axis-1 means back last dim to first
    return X / np.square(X).sum(-1, keepdims=True) ** 0.5   

# ===================================
# Toy datasets of roughly 150 samples
# ===================================

def get_mnist():
    X, y = load_digits(return_X_y=True)
    X = _preprocess(X)
    return X, y


def get_iris():
    X, y = load_iris(return_X_y=True)
    X = _preprocess(X)
    return X, y


def get_wine():
    X, y = load_wine(return_X_y=True)
    X = _preprocess(X)
    return X, y

# =======================================================
# Real-world datasets, load a subset to avoid MemoryError
# =======================================================

def get_fashion_mnist(num_samples: int=10000, num_workers:int = 0):
    train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                             
    ]))
    # total samples: 60000
    loader = torch.utils.data.DataLoader(train_set, batch_size=60000, num_workers=num_workers)
    for batch in loader:
        X, y = batch[0], batch[1]
        print(X.shape)
        _dim = X.shape[-1]  # 28 x 28
        X = X.reshape(60000, _dim**2)  # flatten
        X = _preprocess(X)
        X = X.numpy()
    return X[:num_samples], y[:num_samples]


def get_cifar10(num_samples: int=10000, num_workers:int = 0):
    # total training samples: 50000
    train_set = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10',
        train=True,
        download=True, 
        transform=transforms.Compose([
        transforms.ToTensor()
        ]))
    loader = torch.utils.data.DataLoader(train_set, batch_size=50000, num_workers=num_workers)
    _transform = np.array([0.2989, 0.5870, 0.1140])  # RGB to BW
    for batch in loader:
        X, y = batch[0], batch[1]
        X = X.numpy().swapaxes(1, 2).swapaxes(2,3)
        X = np.array([x.dot(_transform) for x in X])
        _dim = X.shape[-1]  # 32 x 32
        X = X.reshape(50000, _dim**2)  # flatten
        X = _preprocess(X)
    return X[:num_samples], y[:num_samples]


def get_faces_in_wild(min_faces_per_person: int=70, resize: float=0.4):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)  # 9.5mb 
    X = lfw_people.data  # shape: (1288, 1850)
    y = lfw_people.target  # 7 labels
    X = _preprocess(X)
    # label names: ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush', 'Gerhard Schroeder' 'Hugo Chavez' 'Tony Blair']
    return X, y


def get_newsgroup_vectors():
    X, y = [], []
    with open('./data/newsgroup_embeddings.bin', 'rb') as _file:
        X = pickle.load(_file)  # shape: (11314, 512)
    with open('./data/newsgroup_labels.bin', 'rb') as _file:
        y = pickle.load(_file)  # 20 labels
    X = _preprocess(X)
    return X, y



