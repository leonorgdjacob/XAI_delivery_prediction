import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Caminho para os dados
data_path = Path("C:/Users/anale/OneDrive/Documentos/Universidade/5º ANO/TESE/image-dataset")
image_path = data_path / "dataset_images_cv_3/Head_"
test_dir = image_path / "test"

# Transformação para teste (escala de cinza e resize)
test_transform = transforms.Compose([
    transforms.functional.rgb_to_grayscale,  # Converter para escala de cinza
    transforms.Resize((80, 80)),
    transforms.ToTensor()
])

# Dataset e DataLoader de teste
test_data_simple = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=1, shuffle=False)  #le as imagens 1 a 1 -> batchsize
class_names = test_data_simple.classes
print(class_names)

# Carregar o modelo ResNet18
model = models.resnet18(weights='DEFAULT')
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Binario - vaginal e cesariana
model.load_state_dict(torch.load('repositorio-resnet/resnet18-3splits-head/cv3/best-model.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Função Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        # Usando register_full_backward_hook
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_index=None):
        # Forward
        output = self.model(input_tensor)

        # Backward para a classe-alvo
        if class_index is None:
            class_index = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, class_index].backward()

        # Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3))  # Peso global
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]
        cam = torch.clamp(cam, min=0)
        cam = cam / cam.max()  # Normalizar
        return cam.cpu().detach().numpy(), class_index  # Retorna o Grad-CAM e o índice da classe

# Camada alvo (última camada convolucional da ResNet18)
target_layer = model.layer4[-1]

# Inicializar Grad-CAM
grad_cam = GradCAM(model, target_layer)

# Função de visualização com a classe prevista
def visualize_grad_cam(image, cam, predicted_class_idx, image_name):
    # Redimensionar Grad-CAM para o tamanho da imagem
    cam_resized = cv2.resize(cam, (image.shape[2], image.shape[1]))

    # Normalizar para o intervalo [0, 255]
    cam_resized = cam_resized - cam_resized.min()
    cam_resized = cam_resized / cam_resized.max() * 255
    cam_resized = cam_resized.astype(np.uint8)

    # Aplicar mapa de calor
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    # Converter a imagem para RGB se necessário e sobrepor o mapa de calor
    heatmap = np.float32(heatmap) / 255
    image_rgb = np.transpose(image.numpy(), (1, 2, 0))  # Converte para HxWxC
    cam_overlay = 0.5 * image_rgb + 0.5 * heatmap  # Mistura a imagem com o heatmap

    # Exibir a imagem com Grad-CAM e a classe prevista
    plt.imshow(cam_overlay)
    plt.axis('off')  # Remove o eixo
    plt.title(f"Predicted: {class_names[predicted_class_idx]} - Image: {image_name}")  
    plt.show()

# Testar Grad-CAM em um exemplo do DataLoader
for images, labels in test_dataloader_simple:

    image_path = test_dataloader_simple.dataset.imgs[0][1]  # Obtém o caminho da primeira imagem
    image_name = Path(image_path).name  # para saber qual e a imagem

    images, labels = images.to(device), labels.to(device)
    cam, predicted_class_idx = grad_cam.generate(images, class_index=labels.item())
    visualize_grad_cam(images[1].cpu(), cam, predicted_class_idx, image_name)
    break
