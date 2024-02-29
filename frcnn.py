import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define your FCRNN model class
class FCRNN(nn.Module):
    def __init__(self, num_classes):
        super(FCRNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.gru = nn.GRU(32 * 54 * 54, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 54 * 54)
        x = x.unsqueeze(1)  # Add a time dimension
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Get the last output of the GRU
        x = self.fc(x)
        return x

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels, self.classes = self.load_data()

    def load_data(self):
        images = []
        labels = []
        classes = set()  # Using a set to automatically handle unique classes

        # Assuming each subdirectory of root_dir represents a class
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, file_name)
                if not os.path.isfile(image_path):
                    continue
                try:
                    # Attempt to open the file as an image
                    image = Image.open(image_path).convert('RGB')
                    images.append(image_path)
                    labels.append(class_name)  # Use class name as label
                    classes.add(class_name)  # Add class to set of classes
                except Exception as e:
                    print(f"Error opening image file {image_path}: {e}")

        return images, labels, list(classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        # Open image and apply transformations if available
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Map class name to class index
        label_idx = self.classes.index(label)

        return image, label_idx

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your dataset
root_dir = r"C:\Users\nirma\OneDrive\Desktop\models\NASAL_BONE.v4i.yolov5pytorch\train"
dataset = CustomDataset(root_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = FCRNN(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
ap_list = []
confusion_matrices = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    true_labels = []
    predicted_labels = []

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Collect true and predicted labels for each batch
        true_labels.extend(labels.tolist())
        predicted_labels.extend(outputs.argmax(dim=1).tolist())

    # Calculate metrics for the epoch
    from sklearn.metrics import precision_score
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, average_precision_score

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    ap = average_precision_score(true_labels, predicted_labels, average='macro')
    confusion_matrices.append(confusion_matrix(true_labels, predicted_labels))

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    ap_list.append(ap)

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AP: {ap}")

# Save the trained model
torch.save(model.state_dict(), "rfcnn.pth")
print("Model saved successfully.")

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrices[-1], interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(dataset.classes))
plt.xticks(tick_marks, dataset.classes, rotation=45)
plt.yticks(tick_marks, dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Plotting the precision-recall curve
plt.figure(figsize=(8, 6))
for i in range(len(dataset.classes)):
    precision, recall, _ = precision_recall_curve(true_labels, predicted_labels, pos_label=i)
    plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap_list[i]:0.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()
