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
import sys


def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title("Losses over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(train_accuracies, label='Train Accuracy')
    axs[1].plot(val_accuracies, label='Validation Accuracy')
    axs[1].set_title("Accuracies over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(path)



class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_labels = {}

        # Create a mapping of class labels to integers
        self.class_labels = {}
        class_idx = 0

        # Iterate over sub-directories
        for class_dir in os.listdir(self.root_dir):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                self.class_labels[class_dir] = class_idx
                class_idx += 1

                # Iterate over images in the sub-directory
                for img_filename in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_filename)
                    self.images.append(img_path)
                    self.labels.append(self.class_labels[class_dir])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        # Convert grayscale image to RGB
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))
        if self.transform:
            image = self.transform(image)
        return image, label


def predict_label(model, image_path, class_labels_dict):
    '''
    model: a pre-trained PyTorch model
    image_path: the path to the input image
    class_labels_dict: a dictionary that maps integer labels to string labels
    
    return: the string label associated with the input image
    '''
    # Read the image and resize it to the input size of the model
    image = Image.open(image_path)
    image = transforms.Resize(256)(image)
    image = transforms.CenterCrop(224)(image)
    
    # Transform the image to tensor and add a batch dimension
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    # Pass the image through the model
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # Get the index of the maximum probability and convert it to string label
    label_index = torch.argmax(output).item()
    label = class_labels_dict[label_index]
    
    return label