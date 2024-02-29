import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
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

            if label_line:  # Check if label_line is not empty
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
images_folder = r"C:\Users\nirma\OneDrive\Desktop\models\NASAL_BONE.v4i.yolov5pytorch\train\images"
labels_folder = r"C:\Users\nirma\OneDrive\Desktop\models\NASAL_BONE.v4i.yolov5pytorch\train\labels"
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
