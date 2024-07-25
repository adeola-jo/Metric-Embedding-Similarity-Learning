import numpy as np
import torch
from collections import defaultdict
PRINT_LOSS_N = 100


def train(model, optimizer, loader, device='cuda'):
    """
    Trains the model using the provided optimizer and data loader.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loader (torch.utils.data.DataLoader): The data loader for loading the training data.
        device (str, optional): The device to be used for training. Defaults to 'cuda'.

    Returns:
        float: The mean loss over the training iterations.
    """
    losses = []
    model.train()
    for i, data in enumerate(loader):
        anchor, positive, negative, _ = data
        optimizer.zero_grad()
        loss = model.loss(anchor.to(device), positive.to(device), negative.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        if i % PRINT_LOSS_N == 0:
            print(f"Iter: {i}, Mean Loss: {np.mean(losses):.3f}")
    return np.mean(losses)


def compute_representations(model, loader, identities_count, emb_size=32, device='cuda'):
    """
    Compute representations for each identity in the given data loader using the provided model.

    Args:
        model (torch.nn.Module): The model used to compute the representations.
        loader (torch.utils.data.DataLoader): The data loader containing the samples.
        identities_count (int): The number of unique identities in the data loader.
        emb_size (int, optional): The size of the embedding vector. Defaults to 32.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: A tensor containing the averaged representations for each identity.
    """
    model.eval()
    representations = defaultdict(list)
    for i, data in enumerate(loader):
        anchor, id = data[0], data[-1]
        with torch.no_grad():
            repr = model.get_features(anchor.to(device))
            repr = repr.view(-1, emb_size)
        for i in range(id.shape[0]):
            representations[id[i].item()].append(repr[i])
    averaged_repr = torch.zeros(identities_count, emb_size).to(device)
    for k, items in representations.items():
        r = torch.cat([v.unsqueeze(0) for v in items], 0).mean(0)
        averaged_repr[k] = r / torch.linalg.vector_norm(r)
    return averaged_repr


def make_predictions(representations, r):
    """
    Calculates predictions based on the L2 distance between the representations and a given value.

    Parameters:
    representations (numpy.ndarray): Array of representations.
    r (float): Value to subtract from the representations.

    Returns:
    numpy.ndarray: Array of predictions based on the L2 distance.
    """
    return ((representations - r)**2).sum(1)


def evaluate(model, repr, loader, device):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        repr (Representation): The representation used for predictions.
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        device (torch.device): The device to perform the evaluation on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(loader):
        anchor, id = data
        id = id.to(device)
        with torch.no_grad():
            r = model.get_features(anchor.to(device))
            r = r / torch.linalg.vector_norm(r)
        pred = make_predictions(repr, r)
        top1 = pred.min(0)[1]
        correct += top1.eq(id).sum().item()
        total += 1
    return correct/total