import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

#  Upload Dataset Folder
st.title("Dataset Loader and Image Classification")
uploaded_files = st.file_uploader("Upload your dataset (multiple images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    dataset = []
    for file in uploaded_files:
        image = Image.open(file)
        dataset.append(image)
    st.write(f"Uploaded {len(dataset)} images.")

#  Model Selection
model_option = st.selectbox("Choose a pre-trained model", ["ResNet18", "VGG16", "DenseNet"])
st.write(f"You selected {model_option}")

# Load the selected model
def load_model(model_name):
    if model_name == "ResNet18":
        return models.resnet18(pretrained=True)
    elif model_name == "VGG16":
        return models.vgg16(pretrained=True)
    elif model_name == "DenseNet":
        return models.densenet121(pretrained=True)
    else:
        raise ValueError("Unknown model selected!")

model = load_model(model_option)
model.eval()
st.write(f"{model_option} model loaded successfully!")

# Step 4: Image upload for prediction
uploaded_img = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    # Example class names (replace with actual class names)
    class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7']
    st.write(f'Prediction: {class_names[predicted.item()]}')
