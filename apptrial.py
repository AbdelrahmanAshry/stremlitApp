# -*- coding: utf-8 -*-
"""Deploy.ipynb
Importing Libraries
"""
#pip install streamlit

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from threading import Timer

"""#Deploy App streamlit"""
# Streamlit app
st.title("Image Classification")

# Define the model architecture (replace with your actual architecture)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class SimpleDenseNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleDenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = DenseBlock(64, growth_rate=32, num_layers=4)
        self.trans1 = TransitionLayer(192, 96)
        

        self.block2 = DenseBlock(96, growth_rate=32, num_layers=4)
        self.trans2 = TransitionLayer(224, 112)
        

        self.block3 = DenseBlock(112, growth_rate=32, num_layers=4)
        self.trans3 = TransitionLayer(240, 120) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Function to terminate the Streamlit session
def terminate_session():
    st.write("Session terminated due to no file upload.")
    st.stop()

# Streamlit app
st.title("Image Classification")

# File uploader for .pth files
uploaded_model_file = st.file_uploader("Upload a model file (.pth)", type=["pth"])

# File uploader for image files
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize model variable
model = None

if uploaded_model_file is not None:
#    try:
#        # Try loading the model as a full model
           try:
            model = torch.load(uploaded_model_file, map_location=torch.device('cpu'))
#             model = torch.load(uploaded_model_file)
#            if isinstance(uploaded_model_file , torch.nn.Module):
#                model.eval()#full model
#                st.write("Model loaded successfully!")
#            elif isinstance(uploaded_model_file , dict):
               print("This is a state dictionary.")
 # You'll need to load this into a model architecture
               model = SimpleDenseNet(num_classes=7)  # Define your model architecture first
               model.load_state_dict(uploaded_model_file )
               model.eval()  # Now you can use eval()
#            else: 
#                raise ValueError("Loaded file is not a valid model.")
#        except Exception as e:
#            st.write("Failed to load model as a full model. Trying to load as a state dictionary...")
#            try:
#                model = MyModel()  # Define the model architecture
#                model.load_state_dict(torch.load(uploaded_model_file, map_location=torch.device('cpu')))
#                model.eval()
#                st.write("Model loaded successfully from state dictionary!")
#            except Exception as e:
#                st.error(f"An error occurred while loading the model: {e}")
#                st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
#        st.stop()

# Data transformations
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Handle image upload and classification
if uploaded_img is not None:
    if model is not None:
        # Load the image
        image = Image.open(uploaded_img)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image
        image = data_transforms['val'](image).unsqueeze(0)

        # Move model and input to the device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model = model.to(device)
        #image = image.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Map the output to class names
        class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
        st.write(f'Prediction: {class_names[predicted.item()]}')

        # Update the progress bar to complete
        st.write("Classification completed.")
    else:
        st.warning("Please upload a model file first.")
else:
    st.warning("Please upload an image to classify.")
