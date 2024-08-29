import streamlit as st
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import DenseNet121_Weights, VGG16_Weights, ResNet18_Weights
import zipfile
import random
import os
import tempfile

st.title("Image Classification")
# Data transformations
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}
# Load pre-trained models
def load_model(model_name, num_classes):
    if model_name == "ResNet18":
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG16":
        weights = VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "DenseNet":
        weights = DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown model selected!")
    return model

# Check if weights are compatible with the model
def check_weights_compatibility(model, weights_path):
    try:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
        return True
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return False

# Upload model weights
uploaded_weights = st.file_uploader("Upload a model weights file (.pth)", type=["pth"])

if uploaded_weights is not None:
    # Model Selection
    model_option = st.selectbox("Choose a pre-trained model", ["ResNet18", "VGG16", "DenseNet"])
    st.write(f"You selected {model_option}")

    num_classes = 7
    model = load_model(model_option, num_classes)
    st.write(f"{model_option} model loaded successfully!")

    # Verify if uploaded weights are compatible
    if check_weights_compatibility(model, uploaded_weights):
        st.success("Weights loaded successfully!")
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        class_names = ['Cas', 'Cos', 'Gum', 'MC', 'OC', 'OLP', 'OT']
        image = None
        # Upload image for training
        image = st.file_uploader("Upload a picture  file (.jpg)", type=["jpg"])
        if image is not None:
            # Preprocess the image
            image = data_transforms['val'](image).unsqueeze(0)
            with torch.no_grad():
              output = model(image)
              _, predicted = torch.max(output, 1)
              st.write(f'Prediction: {class_names[predicted.item()]}')
              st.write("Classification completed.")
        else:
            st.warning("Please choose or upload an image to classify.")
    else:
        st.error("Invalid model weights provided.")
else:
    st.warning("Please upload a model weights file.")
