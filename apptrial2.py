import streamlit as st
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import  DenseNet121_Weights,VGG16_Weights,ResNet18_Weights
import zipfile
import random
import os
import tempfile

st.title("Image Classification")
# Create a temporary directory to store the uploaded dataset
with tempfile.TemporaryDirectory() as tmp_dir:
    # Allow the user to upload a ZIP file containing the dataset
    uploaded_file = st.file_uploader("Upload a ZIP file containing the dataset", type=["zip"])
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

    
    if uploaded_file is not None:
        # Extract the ZIP file into the temporary directory
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # Identify the root directory containing the images
        def find_root_dir(directory):
            # Traverse the directory to find the actual root of the dataset
            for root, dirs, files in os.walk(directory):
                if len(dirs) == 7 and all(os.path.isdir(os.path.join(root, d)) for d in dirs):
                    return root
            return directory

        # Locate the main dataset directory
        main_dir = find_root_dir(tmp_dir)
        
        # Check if Training, Validation, Testing directories exist
        train_dir = os.path.join(main_dir, "Training")
        val_dir = os.path.join(main_dir, "Validation")
        test_dir = os.path.join(main_dir, "Testing")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
            # If directories aren't found, assume the dataset needs splitting
            full_dataset = datasets.ImageFolder(main_dir)
            
            # Split dataset into train, val, and test
            train_indices, temp_indices = train_test_split(
                list(range(len(full_dataset))), test_size=0.4, stratify=full_dataset.targets)
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, stratify=[full_dataset.targets[i] for i in temp_indices])
            
            # Creating subsets
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            # Apply validation transform to validation and test datasets
            val_dataset.dataset.transform = data_transforms['val']
            test_dataset.dataset.transform = data_transforms['val']
            test_dataset.dataset.transform = data_transforms['train']
            
            st.success("Dataset split into train, validation, and test sets successfully!")
       
        else:            
            # Load the datasets using ImageFolder
            train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
            val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
            test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])
            
            st.success("Datasets loaded successfully!")

        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Display class names
        class_names = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
        st.write(f"Class names: {class_names}")
        
        # Model Selection
        model_option = st.selectbox("Choose a pre-trained model", ["ResNet18", "VGG16", "DenseNet"])
        st.write(f"You selected {model_option}")

        # Load the selected model
        def load_model(model_name):
            if model_name == "ResNet18":
                weights = ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer

            elif model_name == "VGG16":
                weights=VGG16_Weights.DEFAULT
                model = models.vgg16(weights=weights)
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)  # Modify the final layer
            elif model_name == "DenseNet":
                weights = DenseNet121_Weights.DEFAULT
                model = models.densenet121(weights=weights)
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)  # Modify the final layer
            else:
                raise ValueError("Unknown model selected!")
            return   model
        model = load_model(model_option)
        model.eval()
        num_classes = 7
        num_epochs = 10
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer for the classes
        st.write(f"{model_option} model loaded successfully!")
        # User choice: Random image or Upload
        option = st.radio("Select an option", ('Random image from Test dataset', 'Upload an image'))

        # Initialize variable to store the image
        image = None

        # Option 1: Random Image from Dataset
        if option == 'Random image from Test dataset':
            # Pick a random image
            random_index = random.randint(0, len(test_loader) - 1)
            image, label = test_loader.dataset.dataset[test_loader.dataset.indices[random_index]]
            # Convert the tensor image to a PIL image
            Img = transforms.ToPILImage()(image)
            st.image(Img, caption=f'Random Image from Class: {class_names[label]}', use_column_width=True)

        # Option 2: Upload an Image
        elif option == 'Upload an image':
            uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                image = Image.open(uploaded_img)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                image = data_transforms['val'](image)

        # If an image is available, make a prediction
        if image is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
