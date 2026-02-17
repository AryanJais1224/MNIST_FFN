# 1. Imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load and Preprocess Data
data = np.loadtxt("sample_data/mnist_train_small.csv", delimiter=",")

labels = data[:, 0]
images = data[:, 1:]

images = images / 255.0

X = torch.tensor(images, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# 3. Shift Test Images
def shift_images(X, max_shift=10):
    X_shifted = X.clone()

    for i in range(X.shape[0]):
        img = X[i].reshape(28, 28)
        shift = np.random.randint(-max_shift, max_shift + 1)
        img_shifted = torch.roll(img, shifts=shift, dims=1)
        X_shifted[i] = img_shifted.reshape(-1)

    return X_shifted


X_test_shifted = shift_images(X_test, max_shift=10)


# 4. Create Datasets and Loaders
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test_shifted, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 5. Model Definition
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)


# 6. Training Function
def train_model(model, train_loader, test_loader, epochs=50, lr=0.01):

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        epoch_test_acc = 100 * correct / total
        test_acc.append(epoch_test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2f}% | "
              f"Shifted Test Acc: {epoch_test_acc:.2f}%")

    return train_losses, train_acc, test_acc


# 7. Run Training
if __name__ == "__main__":

    model = MNISTNet()
    losses, trainAcc, testAcc = train_model(model, train_loader, test_loader)

    fig, ax = plt.subplots(1, 2, figsize=(14,5))

    ax[0].plot(losses)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(trainAcc, label="Train")
    ax[1].plot(testAcc, label="Shifted Test")
    ax[1].set_title(f"Final Shifted Test Accuracy: {testAcc[-1]:.2f}%")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].legend()

    plt.show()
