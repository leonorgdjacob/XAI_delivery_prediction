import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vision_transformer
from pathlib import Path
import os
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
import torch.nn.functional as F
import json
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from tqdm.auto import tqdm
import shutil
import torchvision.utils
import sys 
from collections import Counter
from torchvision.datasets import ImageFolder
import random
from PIL import Image

sys.path.insert(0,"/home/scarlett/Documents/obstetrics_Carolina/project/MedViT/")
from MedViT_model import MedViT_small, MedViT_base, MedViT_large

# Set the device      
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# novo directory onde vou guardar os resultados 
new_directory = 'medVit-base-pretrained-cv2abdomen'
os.makedirs(new_directory, exist_ok=True)

# Setup path to data folder
data_path = Path("./data/")
image_path = data_path / "cv2-Abdomen"

# Setup train and testing paths
train_dir = image_path / "train"
validation_dir = image_path / "validation"
test_dir = image_path / "test"


# 1. Load original train dataset (to inspect class distribution)
train_data_original = ImageFolder(root=train_dir)

class_names =train_data_original.classes
print(class_names)
NUM_CLASSES = len(class_names) 



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Grayscale(num_output_channels=3), # Converter para escala de cinza
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Grayscale(num_output_channels=3), # Converter para escala de cinza 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])




# Turn image folders into Datasets
train_data_augmented = ImageFolder(root=train_dir, 
                                   transform=train_transform)
validation_data_simple = datasets.ImageFolder (root=validation_dir,
                                             transform=test_transform)
test_data_simple = datasets.ImageFolder (root=test_dir,
                                             transform=test_transform)


# Turn Datasets into DataLoader's
BATCH_SIZE = 16


train_dataloader_augmented = DataLoader (train_data_augmented,
                                         batch_size=BATCH_SIZE,
                                         shuffle= True)
test_dataloader_simple = DataLoader (test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)

validation_dataloader_simple = DataLoader (validation_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)



pretrained_weights_path = '/home/scarlett/Documents/obstetrics_Carolina/project/MedViT/MedViT_base_im1k.pth'
model = MedViT_base(pretrained=True, pretrained_cfg = pretrained_weights_path, num_classes = NUM_CLASSES).cuda()


#model_summary = summary(model,input_size=[1,3,224, 224])
#print(model_summary)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs=500

# Training loop

def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs):
    # Lists to store metrics
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    best_validation_accuracy = 0.0
    best_epoch = 0 
    for epoch in range(num_epochs):
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                validation_loss += loss.item()
                _, predicted_validation = torch.max(outputs, 1)
                correct_validation += (predicted_validation == labels).sum().item()
                total_validation += labels.size(0)
                predictions_validation.extend(predicted_validation.cpu().numpy())
                targets_validation.extend(labels.cpu().numpy())
        validation_losses.append(validation_loss / len(val_loader))
        validation_accuracy = correct_validation / total_validation
        validation_accuracies.append(validation_accuracy)

      
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"validation Loss: {validation_losses[-1]:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

        #Update best model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), 'medVit-base-pretrained-cv2abdomen/best-model.pth')
        

    print(f"Best model saved after epoch: {best_epoch}")  
    torch.save(model.state_dict(), 'medVit-base-pretrained-cv2abdomen/final-model.pth')

    # Calculate confusion matrix for train data 
    train_conf_matrix = confusion_matrix(targets_train, predictions_train)

    # Calculate confusion matrix for validation data 
    validation_conf_matrix = confusion_matrix(targets_validation, predictions_validation)
    
    # Return results
    return train_losses, validation_losses, train_accuracies, validation_accuracies, train_conf_matrix, validation_conf_matrix, best_epoch

train_losses, validation_losses, train_accuracies, validation_accuracies, train_conf_matrix, validation_conf_matrix, best_epoch = train (model, train_dataloader_augmented, validation_dataloader_simple, loss_fn, optimizer, num_epochs)
history ={
    "Train Losses": train_losses,
    "validation Losses": validation_losses,
    "Train Accuracies": train_accuracies,
    "validation Accuracies": validation_accuracies,
    "Best model saved after epoch": best_epoch
}

history_file_path = os.path.join(new_directory, 'history.json')

with open(history_file_path, 'w') as history_file:
    json.dump(history, history_file,indent=1)


plt.figure(figsize=(15, 7))
# Plot train and test loss vs epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss ')
plt.legend()

# Plot train and test accuracy vs epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation Accuracy ')
plt.legend()
plt.savefig(os.path.join(new_directory,'train_validation_loss_accuracy.png'))
plt.show()

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
    '''tp_dir = os.path.join(output_directory, "tp")
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
        shutil.copy(image_path, fn_dir)'''

    return balanced_accuracy, precision, recall, f1, test_conf_matrix, roc_auc, tp, fp, fn, tn, acc, mcc

output_directory = 'medVit-base-pretrained-cv2abdomen/results-bestmodel'
os.makedirs(output_directory, exist_ok=True)
best_model=model
best_model.load_state_dict(torch.load('medVit-base-pretrained-cv2abdomen/best-model.pth'))
best_model.to(device)
balanced_accuracy, precision, recall, f1, test_conf_matrix, roc_auc, tp, fp, fn, tn, acc, mcc = evaluate_model (best_model, test_dataloader_simple,output_directory)
print(mcc)
metrics = {
        "Balanced Accuracy": balanced_accuracy,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "roc auc": roc_auc,
        "matthews_corrcoef": mcc}


metrics_file_path = os.path.join(new_directory, 'metrics_best_model.json')


with open(metrics_file_path, 'w') as metrics_file:
    json.dump(metrics, metrics_file,indent=1)


plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Best Test Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig(os.path.join(new_directory,'best_test_confusion_matrix.png'))
plt.show()

output_directory = 'medVit-base-pretrained-cv2abdomen/results_final_model'
os.makedirs(output_directory, exist_ok=True)
final_model=model
final_model.load_state_dict(torch.load('medVit-base-pretrained-cv2abdomen/final-model.pth'))
final_model.to(device)
balanced_accuracy, precision, recall, f1, final_test_conf_matrix, roc_auc, tp, fp, fn, tn, acc, mcc = evaluate_model (final_model, test_dataloader_simple,output_directory)

metrics = {
        "Balanced Accuracy": balanced_accuracy,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "roc auc": roc_auc,
        "matthews_corrcoef": mcc}


metrics_file_path = os.path.join(new_directory, 'metrics_final_model.json')

with open(metrics_file_path, 'w') as metrics_file:
    json.dump(metrics, metrics_file,indent=1)

plt.figure(figsize=(8, 6))
sns.heatmap(final_test_conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Final Test Confusion Matrix ")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig(os.path.join(new_directory,'final_test_confusion_matrix.png'))
plt.show()