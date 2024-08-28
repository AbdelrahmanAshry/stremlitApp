import streamlit as st
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from PIL import Image
import random
import os
import tempfile
st.title("Image Classification")
# Create a temporary directory to store the uploaded dataset
with tempfile.TemporaryDirectory() as tmp_dir:
    # Allow the user to upload a ZIP file containing the dataset
    uploaded_file = st.file_uploader("Upload a ZIP file containing the dataset", type=["zip"])

    if uploaded_file is not None:
        # Extract the ZIP file into the temporary directory
        import zipfile
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # Assuming the ZIP file contains Training, Validation, Testing directories
        train_dir = os.path.join(tmp_dir, "Training")
        val_dir = os.path.join(tmp_dir, "Validation")
        test_dir = os.path.join(tmp_dir, "Testing")

        # Data preprocess
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

        # Load the datasets using ImageFolder
        train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
        test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms['val'])

        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Display class names
        class_names = train_dataset.classes
        st.write(f"Class names: {class_names}")

        # Model Selection
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
        num_classes = 7
        num_epochs = 10
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for the classes
        st.write(f"{model_option} model loaded successfully!")

        # Train model on Loaded Data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model = torch.compile(model, mode="reduce-overhead")
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            train_accuracy = 100. * correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_accuracy = 100. * correct / total

            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            # Save the model if the validation loss decreases
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'Model saved with validation loss: {val_loss:.4f}')
        
        print('Finished Training')

        # User choice: Random image or Upload
        option = st.radio("Select an option", ('Random image from training dataset', 'Upload an image'))

        # Initialize variable to store the image
        image = None

        # Option 1: Random Image from Dataset
        if option == 'Random image from Test dataset':
            # Pick a random image
            random_index = random.randint(0, len(test_loader) - 1)
            image, label = test_loader.dataset[random_index]
            st.image(transforms.ToPILImage()(image), caption=f'Random Image from Class: {test_loader.dataset.classes[label]}', use_column_width=True)

        # Option 2: Upload an Image
        elif option == 'Upload an image':
            uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                image = Image.open(uploaded_img)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                image = data_transforms['val'](image)

        # If an image is available, make a prediction
        if image is not None:
            model = model.to(device)
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            class_names = ['Cas', 'Cos', 'Gum', 'MC', 'OC', 'OLP', 'OT']
            st.write(f'Prediction: {class_names[predicted.item()]}')
            st.write("Classification completed.")
        else:
            st.warning("Please choose or upload an image to classify.")

