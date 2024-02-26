import os
from matplotlib import patches, pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Define your ResNet model
class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []

        label_files = os.listdir(self.labels_folder)
        for file in label_files:
            label_path = os.path.join(self.labels_folder, file)
            image_path = os.path.join(self.images_folder, file.replace('.txt', '.jpg')) 
            
            with open(label_path, 'r') as f:
                label_line = f.readline().strip()

            label_value = float(label_line.split()[0])
            
            image = Image.open(image_path)
            images.append(image)
            labels.append(label_value)

        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your dataset
images_folder = "C:/Users/My pc/Downloads/collegeproject-master/collegeproject-master/train/images"
labels_folder = "C:/Users/My pc/Downloads/collegeproject-master/collegeproject-master/train/labels"
dataset = CustomDataset(images_folder, labels_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = ResNetModel(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Calculate accuracy on the training set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Training Accuracy: {100 * correct / total}%")

# Save the trained model
torch.save(model.state_dict(), "s.pth")

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    # Calculate intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate area of each bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Example ground truth and predicted bounding boxes
ground_truth_boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]
predicted_boxes = [[60, 60, 170, 170], [180, 180, 280, 280]]

# Calculate IoU for each pair of boxes
ious = [calculate_iou(gt_box, pred_box) for gt_box in ground_truth_boxes for pred_box in predicted_boxes]
print("IoU:", ious)

# Calculate precision, recall, and F1 score
y_true = [1, 0, 1, 0]  # Example ground truth labels
y_pred = [1, 0, 1, 1]  # Example predicted labels
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate Average Precision (AP) and Mean Average Precision (mAP)
y_true = np.array([1, 0, 1, 0])  # Example ground truth labels
y_scores = np.array([0.9, 0.8, 0.3, 0.2])  # Example predicted scores
average_precision = average_precision_score(y_true, y_scores)
print("Average Precision:", average_precision)

# Calculate Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Precision x Recall Curve with 11-point interpolation
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision x Recall Curve')
plt.show()

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()

# Plot bounding boxes on images with IoU scores
fig, ax = plt.subplots(1, len(ground_truth_boxes), figsize=(10, 5))
for i, gt_box in enumerate(ground_truth_boxes):
    ax[i].imshow(np.ones((300, 300, 3)))  # Dummy image
    ax[i].add_patch(patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2] - gt_box[0], gt_box[3] - gt_box[1], linewidth=2, edgecolor='g', facecolor='none'))
    for j, pred_box in enumerate(predicted_boxes):
        ax[i].add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1], linewidth=2, edgecolor='r', facecolor='none'))
        ax[i].text(pred_box[0], pred_box[1], f'IoU: {ious[i*len(predicted_boxes)+j]:.2f}', color='r', fontsize=8)
plt.show()
