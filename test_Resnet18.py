import os
import torch
import torchvision.transforms as transforms
from PIL import Image
# from s import ResNetModel  # Importing the ResNetModel from s.py
from resnet import ResNetModel
# Define the test image path with your new address
test_image_path = r"C:\Users\nirma\OneDrive\Desktop\models\NASAL_BONE.v4i.yolov5pytorch\test\images\a-5-_png_jpg.rf.3c8520af21b50c3d5c7582881abdb6d2.jpg"  # Assuming the test image is in the same folder

# Define a transform to normalize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the trained model
model = ResNetModel()
model.load_state_dict(torch.load("s.pth"))  # Assuming the trained model weights are saved in s.py
model.eval()

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
