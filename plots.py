import csv
import matplotlib.pyplot as plt


# FILENAME = 'results/training_log.csv'
FILENAME = 'results/training_log_3b.csv'

if __name__ == '__main__':
    # Read the CSV file
    with open(FILENAME, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = list(reader)

    # Convert the data to float
    data = [[float(x) for x in row] for row in data]

    # Separate the data into different lists
    epochs, mean_losses, train_accs, test_accs, epoch_times = zip(*data)

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # Plot Mean Loss
    axs[0].plot(epochs, mean_losses, color='tab:red', marker='o')
    axs[0].set_ylabel('Mean Loss')
    axs[0].grid(True)

    # Plot Train Top1 Acc
    axs[1].plot(epochs, train_accs, color='tab:blue', marker='o')
    axs[1].set_ylabel('Train Top1 Acc')
    axs[1].grid(True)

    # Plot Test Accuracy
    axs[2].plot(epochs, test_accs, color='tab:green', marker='o')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid(True)

    # Plot Epoch Time
    axs[3].plot(epochs, epoch_times, color='tab:purple', marker='o')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Epoch Time')
    axs[3].grid(True)

    fig.tight_layout()
    plt.show()


