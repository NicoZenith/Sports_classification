import torch
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
opt, unknown = parser.parse_known_args()
print(opt)

outf = opt.outf

# Define any image preprocessing steps you want to apply
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# Create an instance of the dataset
test_dataset = ImageDataset('test', transform=transform)

class_labels_dict = {v: k for k, v in test_dataset.class_labels.items()}
print(class_labels_dict)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
store_test_acc = []


# Define the device to run the model on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet50 model
model = efficientnet_b3(pretrained=True).to(device)
# Replace the last layer with a custom layer with the number of outputs equal to the number of classes in your dataset
num_classes = len(set(test_dataset.labels))
model.fc = nn.Linear(in_features=2048, out_features=num_classes).to(device)

checkpoint_dir = 'checkpoints'
# Load the state dictionary from the saved file
state_dict = torch.load(checkpoint_dir + '/' + outf + '_trained.pth')
# Load the state dictionary into the model
model.load_state_dict(state_dict)
# Set the model to evaluation mode
model.eval()




for i, (test_images, test_labels) in enumerate(test_dataloader, 0):
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    # Forward pass
    test_outputs = model(test_images)
    test_acc = compute_acc(test_outputs, test_labels)

    store_test_acc.append(test_acc)

avg_test_accuracy = np.mean(store_test_acc)
print("The test accuracy is: ", avg_test_accuracy)






