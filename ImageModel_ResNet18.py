import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
import os
import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import seaborn as sns
import torchinfo
from torchinfo import summary
import json
import shutil
from torchview import draw_graph


# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Setup path to data folder
data_path = Path("image-dataset")
image_path = data_path / "dataset_images_cv_3/Head_"

# Setup train and testing paths
train_dir = image_path / "train"
validation_dir = image_path / "validation"
test_dir = image_path / "test"


# Create tranform with data augmentation 
# Create training transform with TrivialAugment

train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.functional.rgb_to_grayscale,  # Converter para escala de cinza
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.functional.rgb_to_grayscale,  # Converter para escala de cinza
    transforms.Resize((80, 80)),
    transforms.ToTensor()
])


# 2. Create train and test Dataset and DataLoader
# Turn image folders into Datasets

train_data_augmented = datasets.ImageFolder (root=train_dir,
                                             transform=train_transform_trivial_augment)
validation_data_simple = datasets.ImageFolder (root=validation_dir,
                                             transform=test_transform)
test_data_simple = datasets.ImageFolder (root=test_dir,
                                             transform=test_transform)


# Turn Datasets into DataLoader's
BATCH_SIZE = 32


train_dataloader_augmented = DataLoader (train_data_augmented,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
test_dataloader_simple = DataLoader (test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
validation_dataloader_simple = DataLoader (validation_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)

class_names = train_data_augmented.classes
print(class_names)

# Can also get class names as a dictionary
class_dict = train_data_augmented.class_to_idx
print(class_dict)

# Check the lengths of our dataset
print(len(train_data_augmented), len(test_data_simple), len(validation_data_simple))

NUM_CLASSES = len(class_names) 



## DEFINE THE MODEL

# Load the ResNet model
model = models.resnet18(pretrained=False)
# Modificar a primeira camada convolucional para aceitar imagens em escala de cinza
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) 

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.to(device)
num_epochs = 500

model_summary = summary(model,input_size=[1,1,80, 80])
print(model_summary)

model_graph_1 = draw_graph(
    model, input_size=[1,1,80,80],
    graph_name='SimpleCNN',
    hide_inner_tensors=True,
    hide_module_functions=True,
    expand_nested=True
)

model_graph_1.visual_graph
#model_graph_1.resize_graph(scale=0.5)
#model_graph_1.visual_graph



## TRAINING MODEL

def train_and_validation(model, train_loader, validation_loader, optimizer, loss_fn, epochs):
    # Send model to device
    model.to(device)
    
    # Lists to store metrics
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    best_validation_accuracy = 0.0
    best_epoch = 0 

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        predictions_train = []
        targets_train = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)
            predictions_train.extend(predicted_train.cpu().numpy())
            targets_train.extend(labels.cpu().numpy())
        train_losses.append(train_loss / len(train_loader))
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # validation
        model.eval()
        validation_loss = 0.0
        correct_validation = 0
        total_validation = 0
        predictions_validation = []
        targets_validation = []
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                validation_loss += loss.item()
                _, predicted_validation = torch.max(outputs, 1)
                correct_validation += (predicted_validation == labels).sum().item()
                total_validation += labels.size(0)
                predictions_validation.extend(predicted_validation.cpu().numpy())
                targets_validation.extend(labels.cpu().numpy())
        validation_losses.append(validation_loss / len(validation_loader))
        validation_accuracy = correct_validation / total_validation
        validation_accuracies.append(validation_accuracy)

       
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"validation Loss: {validation_losses[-1]:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

        #Update best model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), 'repositorio-resnet/resnet18-3splits-head/cv3/best-model.pth')
        
    print(f"Best model saved after epoch: {best_epoch}")  

    torch.save(model.state_dict(), 'repositorio-resnet/resnet18-3splits-head/cv3/final-model.pth')

    # Calculate confusion matrix for train data 
    train_conf_matrix = confusion_matrix(targets_train, predictions_train)

    # Calculate confusion matrix for validation data 
    validation_conf_matrix = confusion_matrix(targets_validation, predictions_validation)
    
    # Return results
    return train_losses, validation_losses, train_accuracies, validation_accuracies, train_conf_matrix, validation_conf_matrix, best_epoch



## TRAIN AND VALIDATION
train_losses, validation_losses, train_accuracies, validation_accuracies, train_conf_matrix, validation_conf_matrix, best_epoch,  = train_and_validation (model,train_dataloader_augmented,validation_dataloader_simple, optimizer, loss_fn, num_epochs)

print("Train Confusion Matrix:")
print(train_conf_matrix)
print("Validation Confusion Matrix:")
print(validation_conf_matrix)

history_cv3 ={
    "Train Losses": train_losses,
    "validation Losses": validation_losses,
    "Train Accuracies": train_accuracies,
    "validation Accuracies": validation_accuracies,
    "Best model saved after epoch": best_epoch
}


history_file_path = os.path.join(new_directory, 'history_cv3.json')

with open(history_file_path, 'w') as history_file:
    json.dump(history_cv3, history_file,indent=1)


plt.figure(figsize=(15, 7))
# Plot train and test loss vs epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss cv3')
plt.legend()

# Plot train and test accuracy vs epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation Accuracy cv3')
plt.legend()
plt.savefig(os.path.join(new_directory,'train_validation_loss_accuracy.png'))
plt.show()


## TEST LOOP
def evaluate_model (model, test_loader, output_directory):
    model.to(device)
    model.eval()

    predictions = []
    targets = []
    prob_scores = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            predictions.extend (predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            prob_scores.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            # Determine TP, TN, FP, FN
            for i in range(len(predicted)):
                true_label = labels[i].cpu().item()
                predicted_label = predicted[i].cpu().item()
                image_path = test_loader.dataset.samples[len(predictions) - len(predicted) + i][0]
                if true_label == 1:
                    if predicted_label == 1:
                        tn_list.append(image_path)
                    else:
                        fn_list.append(image_path)
                else:
                    if predicted_label == 1:
                        fp_list.append(image_path)
                    else:
                        tp_list.append(image_path)

            

    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=1)
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    mcc = matthews_corrcoef (targets, predictions)
    
    #roc_auc = roc_auc_score (targets, predictions)
    # Calcula a área sob a curva ROC (ROC AUC)
    roc_auc = roc_auc_score(targets, prob_scores)
    
    # Calculate confusion matrix for test data 
    test_conf_matrix = confusion_matrix(targets, predictions)

    # Calculate TN, FN, TP, FP
    tp, fp, fn, tn = confusion_matrix(targets, predictions).ravel()
    tpr = tp / (tp+fp)
    tnr = tn / (tn+fn)
    acc = (tp + tn)/(tp+tn+fp+fn)

    # Create directories if not exist
    tp_dir = os.path.join(output_directory, "tp")
    tn_dir = os.path.join(output_directory, "tn")
    fp_dir = os.path.join(output_directory, "fp")
    fn_dir = os.path.join(output_directory, "fn")

    for directory in [tp_dir, tn_dir, fp_dir, fn_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

      # Copy images to corresponding directories
    for image_path in tp_list:
        shutil.copy(image_path, tp_dir)
    for image_path in tn_list:
        shutil.copy(image_path, tn_dir)
    for image_path in fp_list:
        shutil.copy(image_path, fp_dir)
    for image_path in fn_list:
        shutil.copy(image_path, fn_dir)

    return balanced_accuracy, precision, recall, f1, test_conf_matrix, roc_auc, tp, fp, fn, tn, acc, tp_list, tn_list, fp_list, fn_list, mcc


## METRICS BEST MODEL
#nao sei se interessa



