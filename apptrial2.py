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
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Create the dataset
dataset = CustomImageDataset(images=dataset, transform=data_transforms['train'])

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = load_model(model_option)
model.eval()
num_classes=7
num_epcohs=10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for the classes
st.write(f"{model_option} model loaded successfully!")
#train model on Loaded Data
# Set up the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        labels = ...  # You need to define how to get the labels from the images
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

print('Finished Training')
# Image upload for prediction
uploaded_img = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    image_tensor = data_transforms['val'](image).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    # Example class names (replace with actual class names)
    class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7']
    st.write(f'Prediction: {class_names[predicted.item()]}')
