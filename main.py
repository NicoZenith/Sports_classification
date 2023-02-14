import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b3
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
opt, unknown = parser.parse_known_args()
print(opt)

outf = opt.outf

checkpoint_dir = os.path.join('', "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

results_dir = os.path.join('', "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)




# Define any image preprocessing steps you want to apply
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Create an instance of the dataset
train_dataset = ImageDataset('train', transform=transform)
val_dataset = ImageDataset('valid', transform=transform)
test_dataset = ImageDataset('test', transform=transform)

class_labels_dict = {v: k for k, v in train_dataset.class_labels.items()}

print(class_labels_dict)

# Use the dataset with a DataLoader to load the data in batches
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)


# Define the device to run the model on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet50 model
model = efficientnet_b3(pretrained=True).to(device)


# Replace the last layer with a custom layer with the number of outputs equal to the number of classes in your dataset
num_classes = len(set(train_dataset.labels))
model.fc = nn.Linear(in_features=2048, out_features=num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
show_every = 50


# Define the number of consecutive epochs without improvement to trigger early stopping
patience = 5
# Keep track of the best validation loss
best_val_loss = float('inf')
# Keep track of the number of epochs without improvement
no_improvement_epochs = 0


# Train the model
for epoch in range(100):  # number of epochs

    store_train_loss = []
    store_train_acc = []
    store_val_loss = []
    store_val_acc = []

    for i, (images, labels) in enumerate(train_dataloader, 0):
        images = images.to(device)
        labels = labels.to(device)
        # One-hot encode the labels
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_acc = compute_acc(outputs, labels)

        # Print the statistics
        store_train_loss.append(loss.item())
        store_train_acc.append(train_acc)

        if i % show_every == 0:    # print every 200 mini-batches
            print('[%d, %5d] Training loss: %.3f  Training acc: %.3f' % (epoch + 1, i + 1, np.mean(store_train_loss[-show_every:] 
            ) , np.mean(store_train_acc[-show_every:]) ))


    # compute epoch loss and accuracy 
    train_losses.append(np.mean(store_train_loss))
    train_accuracies.append(np.mean(store_train_acc))

    # Evaluate the model on the validation set 
    model.eval()
    for i, (val_images, val_labels) in enumerate(val_dataloader, 0):
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        # Forward pass
        val_outputs = model(val_images)
        val_loss = criterion(val_outputs, val_labels)
        val_acc = compute_acc(val_outputs, val_labels)

        # Print the statistics
        store_val_loss.append(val_loss.item())
        store_val_acc.append(val_acc)

    mean_val_loss = np.mean(store_val_loss)
    # Check if validation loss has improved
    if np.mean(mean_val_loss < best_val_loss):
        best_val_loss = mean_val_loss
        no_improvement_epochs = 0
        # Save the model's state
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, outf + "_trained.pth"))
    else:
        no_improvement_epochs += 1


    # compute epoch loss and accuracy 
    val_losses.append(np.mean(store_val_loss))
    val_accuracies.append(np.mean(store_val_acc))

    # Print loss and acc at the end of the epoch
    print("Epoch {}: Train Loss: {:.4f}, Validation Loss: {:.4f}, Train Accuracy: {:.2f}%, Validation Accuracy: {:.2f}%".format
    (epoch, train_losses[-1], val_losses[-1], train_accuracies[-1], val_accuracies[-1]))

    # Save figure 
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, results_dir + '/' + outf + "_loss_accuracies.pdf")

print('Finished Training')
