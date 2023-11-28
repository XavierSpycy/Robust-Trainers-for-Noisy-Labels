import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple

class Dataset(object):
    """
    Dataset class for loading data from the given path.
    """

    def __init__(self, file_name: str, root: str='datasets') -> None:
        """
        Initialize the dataset class.
        The base folder is set to 'datasets'.

        Parameters:
        file_name (str): The name of the file to load.
        root (str): The root directory where datasets are stored.
        """
        self.root = os.path.join(root, file_name)
        self.Str = None
        self.T = None
        self.name = file_name.split('.npz')[0] if '.npz' in file_name else None

    def load_data(self, random_state=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the dataset from the given path.

        Returns:
        Xtr (np.ndarray): The training set.
        Str (np.ndarray): The training set labels.
        Xts (np.ndarray): The test set.
        Yts (np.ndarray): The test set labels.
        """
        dataset = np.load(self.root)
        Xtr, self.Str, Xts, self.Yts = dataset['Xtr'], dataset['Str'], dataset['Xts'], dataset['Yts']
        if len(dataset['Xtr'].shape) == 3:
            Xtr = dataset['Xtr'][..., np.newaxis]
            Xts = dataset['Xts'][..., np.newaxis]
        self.mean = np.mean(Xtr, axis=(0, 1, 2)).tolist()
        self.std = np.std(Xtr, axis=(0, 1, 2)).tolist()
        Xtr, Xval, Str, Sval = train_test_split(Xtr, self.Str, test_size=0.2, random_state=random_state, stratify=self.Str)
        return (Xtr, Str), (Xval, Sval), (Xts, self.Yts)

    def label_distribution(self) -> dict:
        """
        Get the label distribution of the dataset.

        Returns:
        train_label_dist (dict): The label distribution of the training set.
        test_label_dist (dict): The label distribution of the test set.
        """
        if self.Str is None:
            _, self.Str, _, self.Yts = self.load_data()
        unique_train_labels, counts_train_labels = np.unique(self.Str, return_counts=True)
        unique_test_labels, counts_test_labels = np.unique(self.Yts, return_counts=True)
        return dict(zip(unique_train_labels, counts_train_labels)), dict(zip(unique_test_labels, counts_test_labels))

class cifar(Dataset):
    def __init__(self) -> None:
        """
        Intialize the dataset class for CIFAR.
        The file name is set to 'CIFAR.npz'.
        """
        super().__init__('CIFAR.npz')

class fashion_mnist_05(Dataset):
    def __init__(self) -> None:
        """
        Initialize the dataset class for Fashion MNIST with 0.5 noise.
        The file name is set to 'FashionMNIST0.5.npz'.
        """
        super().__init__('FashionMNIST0.5.npz')
        self.T = np.array([[0.5, 0.2, 0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]])

class fashion_mnist_06(Dataset):
    def __init__(self) -> None:
        """
        Initialize the dataset class for Fashion MNIST with 0.6 noise.
        The file name is set to 'FashionMNIST0.6.npz'.
        """
        super().__init__('FashionMNIST0.6.npz')
        self.T = np.array([[0.4, 0.3, 0.3],[0.3, 0.4, 0.3],[0.3, 0.3, 0.4]])

class ImageDataset(data.Dataset):
    """
    Image Dataset class for loading data from the given path.
    """
    def __init__(self, images, labels, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_augment=False, one_hot=False, num_classes=3):
        """
        Initialize the image dataset class.

        Parameters:
        images (np.ndarray): The images.
        labels (np.ndarray): The labels.
        mean (list): The mean of the dataset.
        std (list): The standard deviation of the dataset.
        is_augment (bool): Whether to augment the dataset.
        """
        if images.shape[-1] == 1: # Grayscale
            self.images = [Image.fromarray(ndarray.squeeze(-1), mode='L') for ndarray in images]
        elif images.shape[-1] == 3: # RGB
            self.images = [Image.fromarray(ndarray, mode='RGB') for ndarray in images]
        # Convert to tensor
        self.labels = torch.from_numpy(labels)
        if one_hot:
            self.labels = F.one_hot(self.labels, num_classes=num_classes)
        # Define transforms
        if is_augment:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(10),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.ToTensor(),
                v2.Normalize(mean, std)
            ])
        else:
            self.transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean, std)
            ])

    def __len__(self):
        """
        Override the __len__ method to return the size of the dataset

        Returns:
        len (int): The size of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Override the __getitem__ method to return the image and label at the given index.

        Parameters:
        idx (int): The index of the image and label.

        Returns:
        img (torch.Tensor): The image.
        lab (torch.Tensor): The label.
        """
        img = self.images[idx]
        lab = self.labels[idx]
        img = self.transform(img)
        return idx, img, lab

class MaskedDataset(data.Dataset):
    def __init__(self, original_dataset, mask):
        self.original_dataset = original_dataset
        self.mask = mask
        self.clean_indices = [i for i, is_clean in enumerate(mask) if is_clean]

    def __len__(self):
        return len(self.clean_indices)

    def __getitem__(self, idx):
        original_idx = self.clean_indices[idx]
        return self.original_dataset[original_idx]

def get_test_loader(dataset):
    """
    Get the data loaders for the given datasets.
    """
    return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)