import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine, fetch_lfw_people, fetch_20newsgroups_vectorized
import torch
import torchvision
import torchvision.transforms as transforms


def _preprocess(X):
    return X / np.square(X).sum(-1, keepdims=True) ** 0.5   

# ===================================
# Toy datasets of roughly 150 samples
# ===================================

def load_mnist():
    X, y = load_digits(return_X_y=True)
    X = _preprocess(X)
    return X, y


def load_iris():
    X, y = load_iris(return_X_y=True)
    X = _preprocess(X)
    return X, y


def load_wine():
    X, y = load_wine(return_X_y=True)
    X = _preprocess(X)
    return X, y

# =======================================================
# Real-world datasets, load a subset to avoid MemoryError
# =======================================================

def load_fashion_mnist(num_samples: int=10000):
    train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                             
    ]))
    # total samples: 60000
    loader = torch.utils.data.DataLoader(train_set, batch_size=60000, num_workers=-1)
    for batch in loader:
        X, y = batch[0], batch[1]
        _dim = X.shape[-1]  # 28 x 28
        X = X.reshape(num_samples, _dim**2)  # flatten
        X = _preprocess(X)
        X = X.numpy()
    return X[:num_samples], y[:num_samples]


def load_cifar10(num_samples: int=10000):
    train_set = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10',
        train=True,
        download=True, 
        transform=transforms.Compose([
        transforms.ToTensor()
        ]))
    # 60000 samples
    loader = torch.utils.data.DataLoader(train_set, batch_size=60000, num_workers=-1)
    for batch in loader:
        X, y = batch[0], batch[1]
    _dim = X.shape[-1]  # 32 x 32
    X = X.reshape(num_samples, _dim**2)  # flatten
    X = _preprocess(X)
    X = X.numpy()
    return X[:num_samples], y[:num_samples]


def load_faces_in_wild(min_faces_per_person: int=70, resize: float=0.4):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)  # 9.5mb 
    X = lfw_people.data  # shape: (1288, 1850)
    y = lfw_people.target  # 7 labels
    X = _preprocess(X)
    # lfw_people.images.shape  # 50 * 37 = 1850
    # lfw_people.target_names  # 7 people
    # ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush', 'Gerhard Schroeder' 'Hugo Chavez' 'Tony Blair']
    return X, y


def load_newsgroup_vectors():
    # can fit in memory
    news_vectors = fetch_20newsgroups_vectorized()  # 11.7gb
    X = news_vectors.data.toarray()  # shape: (11314, 130107)
    y = news_vectors.target  # 20 labels
    # news_vectors.target_names  # 20 categories
    X = _preprocess(X)
    return X, y



