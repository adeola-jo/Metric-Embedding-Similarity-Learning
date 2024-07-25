from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import os
import matplotlib.pyplot as plt



class MNISTMetricDataset(Dataset):
    """
    A custom dataset class for the MNIST dataset used for metric learning.

    Args:
        root (str, optional): The root directory where the MNIST dataset is stored. Defaults to "../data/MNIST".
        split (str, optional): The split of the dataset to use. Can be one of ['train', 'test', 'traineval']. Defaults to 'train'.
        remove_class (int, optional): The class to remove from the dataset. Defaults to None.

    Attributes:
        root (str): The root directory where the MNIST dataset is stored.
        split (str): The split of the dataset being used.
        images (torch.Tensor): The tensor containing the image data.
        targets (torch.Tensor): The tensor containing the target labels.
        classes (list): The list of classes in the dataset.
        target2indices (defaultdict): A dictionary mapping target labels to the indices of the corresponding images.

    Methods:
        _sample_negative(index): Samples a negative example index for a given anchor index.
        _sample_positive(index): Samples a positive example index for a given anchor index.
        __getitem__(index): Retrieves the data for a given index.
        __len__(): Returns the length of the dataset.
    """

    def __init__(self, root="../data/MNIST", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split

        # Always set download=True. The dataset will only be downloaded if it's not already present in the directory.
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)

        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            self.images = self.images[self.targets != remove_class]
            self.targets = self.targets[self.targets != remove_class]
            self.classes.remove(remove_class)

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        target_id = self.targets[index].item()
        other_classes = self.classes[:]
        other_classes.remove(target_id)
        negative_class = choice(other_classes)
        negative_index = choice(self.target2indices[negative_class])
        return negative_index

    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        positive_indices = self.target2indices[target_id][:]
        positive_indices.remove(index)
        if not positive_indices:  # In case there's only one image in the class
            return index  # Returning the same image as a fallback
        return choice(positive_indices)

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
    

if __name__ == "__main__":
    dataset = MNISTMetricDataset()
    for i in range(10):
        anchor, positive, negative, target = dataset[i]
        plt.subplot(131)
        plt.imshow(anchor.squeeze(0).numpy(), cmap='gray')
        plt.title(f"Anchor: {target}")
        plt.subplot(132)
        plt.imshow(positive.squeeze(0).numpy(), cmap='gray')
        plt.title(f"Positive: {target}")
        plt.subplot(133)
        plt.imshow(negative.squeeze(0).numpy(), cmap='gray')
        plt.title(f"Negative: {target}")
        plt.show()