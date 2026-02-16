"""
MNIST Analysis Script
---------------------
- Loads MNIST CSV data
- Visualizes random digits
- Shows examples of digit 7
- Computes correlation between all 7s
- Displays average 7 image
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    """Load MNIST CSV file."""
    data = np.loadtxt(filepath, delimiter=",")
    labels = data[:, 0]
    images = data[:, 1:]
    return labels, images


def show_random_digits(labels, images):
    """Plot random digits as raw pixel vectors."""
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    for ax in axs.flatten():
        idx = np.random.randint(0, images.shape[0])
        ax.plot(images[idx, :], "k.")
        ax.set_title(f"Label: {int(labels[idx])}")

    plt.suptitle("How the FFN model sees the data", fontsize=16)
    plt.tight_layout()
    plt.show()


def show_example_sevens(labels, images):
    """Display first 12 images of digit 7."""
    sevens = np.where(labels == 7)[0]

    fig, axs = plt.subplots(2, 6, figsize=(12, 5))

    for i, ax in enumerate(axs.flatten()):
        img = images[sevens[i]].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.suptitle("Example 7's", fontsize=16)
    plt.tight_layout()
    plt.show()

    return sevens


def analyze_sevens(images, sevens):
    """Compute correlation and average image for digit 7."""
    seven_data = images[sevens]

    print("Shape of all 7s:", seven_data.shape)

    # Correlation matrix
    C = np.corrcoef(seven_data)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # Correlation heatmap
    im = ax[0].imshow(C, vmin=0, vmax=1)
    ax[0].set_title("Correlation across all 7's")
    plt.colorbar(im, ax=ax[0])

    # Histogram of unique correlations
    unique_corr = np.triu(C, k=1).flatten()
    unique_corr = unique_corr[unique_corr != 0]

    ax[1].hist(unique_corr, bins=100)
    ax[1].set_title("Unique Correlations")
    ax[1].set_xlabel("Correlation")
    ax[1].set_ylabel("Count")

    # Average 7 image
    avg_seven = np.mean(seven_data, axis=0).reshape(28, 28)
    ax[2].imshow(avg_seven, cmap="gray")
    ax[2].set_title("Average of all 7's")

    plt.tight_layout()
    plt.show()


def main():
    labels, images = load_data(filepath)

    print("Labels shape:", labels.shape)
    print("Images shape:", images.shape)

    show_random_digits(labels, images)
    sevens = show_example_sevens(labels, images)
    analyze_sevens(images, sevens)


if __name__ == "__main__":
    main()
