

# Metric Embedding for Similarity Learning

## Project Overview

This project explores the implementation and training of metric embedding models for similarity learning using the MNIST dataset. The goal is to train a model that can embed images into a feature space where similar images are closer together, facilitating tasks such as image retrieval and classification based on similarity.

---

## Project Objectives

1. **Data Loading:**
   - Implement data loading to enable training models for metric embedding with triplet loss. Adapt the MNIST dataset to retrieve anchors, corresponding positive, and negative examples.

2. **Define a Metric Embedding Model:**
   - Implement a model for metric embedding with a triplet loss. Design a convolutional module (BNReLUConv) with group normalization, ReLU activation, and convolution.

3. **Training and Evaluating the Model:**
   - Train the metric embedding model on a subset of the MNIST training set. Perform classification of images from the validation subset and measure the accuracy.

4. **Classification Based on Distances in Image Space:**
   - Implement an IdentityModel for classification based on distances in image space and compare its performance with the metric embedding model.

5. **Data Visualization:**
   - Visualize the arrangement of data in the feature space and image space using principal component analysis (PCA).

---

## Project Structure

- **dataset.py:** Defines the MNISTMetricDataset class for loading the MNIST dataset and sampling positive and negative examples.
- **model.py:** Implements the SimpleMetricEmbedding and IdentityModel classes.
- **utils.py:** Contains utility functions for training, evaluating, and computing representations.
- **task3a.py:** Script for training the metric embedding model on the full MNIST dataset.
- **task3b.py:** Script for training the metric embedding model without one class (class 0).
- **task4.py:** Script for visualizing the data in the feature space and image space using PCA.
- **plots.py:** Script for plotting training logs.
- **tests.py:** Unit tests for various components of the project.
- **lab4.pdf:** Lab guide with detailed instructions and explanations.

---

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```sh
git clone <repository_url>
cd metric-embedding
```

### 2. Set Up the Python Environment

Create a virtual environment to manage dependencies:

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

---

## Usage Instructions

### Task 1: Data Loading

- **File:** `dataset.py`
- **Objective:** Implement the `_sample_positive` and `_sample_negative` methods for sampling positive and negative examples.

### Task 2: Define Metric Embedding Model

- **File:** `model.py`
- **Objective:** Implement the `SimpleMetricEmbedding` model with a triplet loss function and the convolutional module `BNReLUConv`.

### Task 3: Training and Evaluating the Model

#### Part A: Train on Full MNIST Dataset

- **Script:** `task3a.py`
- **Run:**
  ```sh
  python task3a.py
  ```

#### Part B: Train without Class 0

- **Script:** `task3b.py`
- **Run:**
  ```sh
  python task3b.py
  ```

### Task 4: Data Visualization

- **Script:** `task4.py`
- **Run:**
  ```sh
  python task4.py
  ```

### Plotting Training Logs

- **Script:** `plots.py`
- **Run:**
  ```sh
  python plots.py
  ```

### Running Tests

- **Script:** `tests.py`
- **Run:**
  ```sh
  python -m unittest discover
  ```

---

## Notes

- Ensure that the MNIST dataset is available in the specified directories.
- Modify configuration files as needed to suit your experimental setup.
- Detailed instructions and explanations are provided in `lab4.pdf`.

---

## Troubleshooting

- **Dataset Loading Issues:** Verify that the MNIST dataset is correctly downloaded and placed in the specified directory.
- **Model Training Issues:** Ensure your environment is correctly set up and dependencies are installed.
- **Visualization Issues:** Ensure that matplotlib and other plotting libraries are correctly installed.

---

## Example `requirements.txt`

```txt
numpy
torch
torchvision
matplotlib
tqdm
```

---

## Acknowledgements

This project is part of the Deep Learning course at the University of Zagreb, Faculty of Electrical Engineering and Computing (FER). The materials and instructions provided are for educational purposes only and are not publicly available for use without permission.

---
