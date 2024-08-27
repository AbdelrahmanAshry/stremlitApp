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
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Replace with your model architecture

    def forward(self, x):
        return self.model(x)

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

#if uploaded_model_file is not None:
#    try:
#        # Try loading the model as a full model
#        try:
            model = torch.load(uploaded_model_file, map_location=torch.device('cpu'))
            if isinstance(model, torch.nn.Module):
                model.eval()
                st.write("Model loaded successfully!")
            else:
                raise ValueError("Loaded file is not a valid model.")
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
#    except Exception as e:
#        st.error(f"An error occurred while loading the model: {e}")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        image = image.to(device)

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
