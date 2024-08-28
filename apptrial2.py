
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Define the data transforms (using the ones from the simplified DenseNet model)
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Load the model for inference
model.load_state_dict(torch.load('model.pth'))
model.eval()

def predict(image):
    image = data_transforms['val'](image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit app
st.title("Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    predicted_class = predict(image)
    
    # Map the output to class names (define these according to your dataset)
    class_names = ['CaS', 'CoS ', 'Gum', 'MC', 'OC', 'OLP', 'OT']
    st.write(f'Prediction: {class_names[predicted_class]}')
