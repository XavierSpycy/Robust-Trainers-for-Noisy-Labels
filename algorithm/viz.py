import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def display_samples(X, y, sample_size=16, figsize=(6, None)):
    """
    Display a random sample of images from the dataset.

    Parameters:
    X (np.ndarray): The dataset.
    y (np.ndarray): The labels.
    sample_size (int): The number of images to display.
    figsize (tuple): The size of the figure.

    Returns:
    indices (np.ndarray): The indices of the images displayed.
    """
    if len(X.shape) == 3:
        # If the dataset is grayscale, add a channel dimension
        X_ = X[..., np.newaxis]
    else:
        # Otherwise, copy the dataset
        X_ = X.copy()

    # Decide on the layout of the subplots based on the sample_size
    side_length = int(np.ceil(np.sqrt(sample_size)))
    gap = side_length * side_length - sample_size

    rows = side_length
    cols = side_length if gap < side_length else side_length - 1

    # Calculate the width of the figure based on the provided height and the number of columns
    if figsize[1] is None:
        width_per_image = figsize[0] / cols
        width = width_per_image * cols
    else:
        width = figsize[1]

    _, ax = plt.subplots(rows, cols, figsize=(figsize[0], width))

    indices = np.random.choice(len(X_), sample_size, replace=False)

    for i in range(rows * cols):
        row, col = divmod(i, cols)

        # Only plot images for the indices we have
        if i < sample_size:
            idx = indices[i]
            image_data = X_[idx]
            if image_data.shape[2] == 1:
                image_data = image_data.squeeze(-1)
            ax[row, col].imshow(image_data)
            ax[row, col].set_title(y[idx] + 1)
        ax[row, col].set_axis_off()

    plt.tight_layout()
    plt.show()
    return indices

def tSNE(X, y, dataset_name="", perplexity=30.0, random_seed=None):
    """
    Perform t-SNE on the dataset.

    Parameters:
    X (np.ndarray): The dataset.
    y (np.ndarray): The labels.
    dataset_name (str): The name of the dataset.
    perplexity (float): The perplexity.
    random_seed (int): The random seed.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed)
    X_flat = X.reshape((X.shape[0], -1))
    X_tsne = tsne.fit_transform(X_flat)
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title('t-SNE visualization of ' + dataset_name + ' dataset')
    plt.show()

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor given the mean and standard deviation.
    """
    tensor = tensor.clone()  # Avoid modifying the original tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def display_datasets(dataset, mean, std, rows=10, cols=10):
    """
    Display random samples from a PyTorch dataset.

    Parameters:
    - dataset: PyTorch Dataset object
    - rows: number of rows for displaying images
    - cols: number of columns for displaying images
    - figsize: size of the displayed figure
    """
    num_display = rows * cols
    indices = torch.randperm(len(dataset))[:num_display]

    _, axes = plt.subplots(rows, cols, figsize=(rows, cols))
    for idx, ax in zip(indices, axes.flatten()):
        _, image, label = dataset[idx]
        # Denormalize the image
        image = denormalize(image, mean, std)
        image_np = image.permute(1, 2, 0).numpy()

        if image_np.shape[2] == 1:  # If grayscale, remove the channel dimension
            image_np = image_np.squeeze(2)
            cmap = 'gray'
        else:
            cmap = None

        ax.imshow(image_np)
        ax.set_title(f'Label: {label}', fontsize=8, y=.9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()