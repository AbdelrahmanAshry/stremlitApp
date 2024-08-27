# -*- coding: utf-8 -*-
"""Deploy.ipynb
Importing Libraries
"""
#pip install streamlit

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from threading import Timer
#import io
"""#Deploy App streamlit"""

# Function to terminate the Streamlit session
def terminate_session():
    st.write("Session terminated due to no file upload.")
    st.stop()

# File uploader for .pth files
uploaded_model_file = st.file_uploader("Upload a model file (.pth)", type=["pth"])
"""
timer = Timer(60.0, terminate_session)

# Start a timer to terminate the session if no file is uploaded within 40 seconds
if uploaded_model_file is None:
 #   timer.start() """
if uploaded_model_file is not None:
    
    # Load the model
    try:
        model = torch.load(uploaded_model_file)
        model.eval()
        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()  # Stop the app if there is any error
        

# Data transformations
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}
        
# Streamlit app
st.title("Image Classification")
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# Progress bar
progress_bar = st.progress(0)

if uploaded_img is not None:
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
    class_names = ['CaS', 'CoS ', 'Gum', 'MC', 'OC', 'OLP', 'OT']
    st.write(f'Prediction: {class_names[predicted.item()]}')

    # Update the progress bar to complete
    progress_bar.progress(100)
    st.write("Classification completed.")
