# Explainability of Models Applied in Obstetric Medical Imaging

This project explores the implementation of explainable artificial intelligence (XAI) methods to image-based Machine Learning models previously developed to predict the mode of delivery  — vaginal delivery or cesarean section — after induction of labor (IOL). The image data used for model training corresponded to third trimester ultrasound images from three fetal anatomical views (abdomen, head, and femur)
Available model architectures included CNN-based ResNet-18 and DenseNet-169, and Transformer-based MedViT. The three visual XAI techniques employed were backpropagation-based: Grad-CAM, Grad-CAM++, and HiRes-CAM. 

## Usage
* Image Data: code expects .png format for image inputs
* Load Model: recreate the trained model processing and load the corresponding .pth weights
* XAI technique: pick the desired method from the ones available trough pytorch-grad-cam repository folder
* Target Layer: select the model layer to be used for generating the saliency map


## Data Availability
The data is not made publicly available for privacy and ethical reasons.

## Credits
This work builds upon and benefits from code made publicly available by the following repositories:
* ***pytorch-grad-cam* by Jacob Gildenblat** - for explainability methods employment (pytorch-grad-cam folder)
* ***MedViT* by Omid Nejati** - for MedViT model inference (MedViT folder)


