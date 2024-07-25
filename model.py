import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import MNISTMetricDataset


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        """
        Get the features of the input image.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The vectorized features of the image.
        """
        feats = img.view(img.size(0), -1)
        return feats
    
    
class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_maps_in))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    """
    SimpleMetricEmbedding is a PyTorch module that implements a simple metric embedding network.
    It takes an input image and produces a fixed-size embedding vector.

    Args:
        input_channels (int): The number of input channels in the image.
        emb_size (int, optional): The size of the embedding vector. Defaults to 32.

    Attributes:
        emb_size (int): The size of the embedding vector.
        conv1 (nn.Module): The first convolutional layer.
        conv2 (nn.Module): The second convolutional layer.
        conv3 (nn.Module): The third convolutional layer.
        pool (nn.Module): The max pooling layer.
        global_pool (nn.Module): The global average pooling layer.

    Methods:
        get_features(img): Extracts the features from the input image.
        loss(anchor, positive, negative, margin): Calculates the loss for the metric embedding.
    """

    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.conv3 = _BNReluConv(emb_size, emb_size)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def get_features(self, img):
        """
        Extracts the features from the input image.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        x = self.pool(self.conv1(img))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        print(x.size())
        # x = self.global_pool(self.conv3(x))
        x = self.global_pool(x)
        print(x.size())
        
        # print(x.size())
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative, margin=1.0):
        """
        Calculates the loss for the metric embedding.

        Args:
            anchor (torch.Tensor): The anchor image tensor.
            positive (torch.Tensor): The positive image tensor.
            negative (torch.Tensor): The negative image tensor.
            margin (float, optional): The margin for the triplet loss. Defaults to 1.0.

        Returns:
            torch.Tensor: The calculated loss.
        """
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        positive_dist = torch.norm(a_x - p_x, p=2, dim=1)
        negative_dist = torch.norm(a_x - n_x, p=2, dim=1)
        loss = torch.relu(positive_dist - negative_dist + margin)
        return loss.mean()


if __name__ == "__main__":
    dataset = MNISTMetricDataset()
    #create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = SimpleMetricEmbedding(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(100):
        anchor, positive, negative, target = next(iter(dataloader))
        loss = model.loss(anchor, positive, negative)
        import  sys
        sys.exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        # plt.imshow(anchor.squeeze(0).numpy(), cmap='gray')
        # plt.show()
        # plt.imshow(positive.squeeze(0).numpy(), cmap='gray')
        # plt.show()
        # plt.imshow(negative.squeeze(0).numpy(), cmap='gray')
        # plt.show()