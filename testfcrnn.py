
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

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

# Load the saved model
model = FCRNN(num_classes=2)  # Assuming 2 classes (nasal bone fracture and no nasal bone fracture)
model.load_state_dict(torch.load("rfcnn.pth"))
model.eval()

# Define a transform to normalize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the test image path
test_image_path = r"path\to\your\test\image.jpg"

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to make predictions on a single image
def predict_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output).item()
        return predicted_class

# Make prediction on the test image
prediction = predict_image(test_image_path)

# Display the prediction
if prediction == 1:
    print("The image has a nasal bone fracture.")
else:
    print("The image does not have a nasal bone fracture.")
