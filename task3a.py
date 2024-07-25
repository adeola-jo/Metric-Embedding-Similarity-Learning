
import time
import torch.optim
import csv
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False
EVAL_ON_IDENTITY = False

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "../data/MNIST/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 10

    with open('training_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Mean Loss", "Train Top1 Acc", "Test Accuracy", "Epoch Time"])

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t0 = time.time_ns()
            train_loss = train(model, optimizer, train_loader, device)
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
            if EVAL_ON_TEST or EVAL_ON_TRAIN:
                print("Computing mean representations for evaluation...")
                representations = compute_representations(model, train_loader, num_classes, emb_size, device)
            if EVAL_ON_TRAIN:
                print("Evaluating on training set...")
                acc1 = evaluate(model, representations, traineval_loader, device)
                print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
            if EVAL_ON_TEST:
                print("Evaluating on test set...")
                acc1 = evaluate(model, representations, test_loader, device)
                print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
            t1 = time.time_ns()
            epoch_time = (t1-t0)/10**9
            print(f"Epoch time (sec): {epoch_time:.1f}")

            # Write the results to the CSV file
            writer.writerow([epoch, train_loss, acc1, acc1, epoch_time])

    #save the trained model
    torch.save(model.state_dict(), "simple_metric_model.pth")

    #NOTE: EVALUATING IN THE IMAGE SPACE
    if EVAL_ON_IDENTITY:
        identity_model = IdentityModel().to(device)
        emb_size = 28*28
        t0 = time.time_ns()
        print('Evaluating in the image space using the IdentityModel...')
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(identity_model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating identity model on training set...")
            acc1 = evaluate(identity_model, representations, traineval_loader, device)
            print(f"Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating identity model on test set...")
            acc1 = evaluate(identity_model, representations, test_loader, device)
            print(f"Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
