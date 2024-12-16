import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

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
test_dataloader_simple = DataLoader(test_data_simple, batch_size=1, shuffle=False)
class_names = test_data_simple.classes
print(class_names)

# Carregar o modelo ResNet18
model = models.resnet18(weights='DEFAULT')
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Binário - Cesárea e Vaginal
model.load_state_dict(torch.load('repositorio-resnet/resnet18-3splits-head/cv3/best-model.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Dividindo o modelo em feature_net e head_net
feature_net = nn.Sequential(*list(model.children())[:-2])  # Remove as camadas fc e avgpool
head_net = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # Reduz dimensões para (1, 1)
    nn.Flatten(),                  # Achata para tensor 2D
    model.fc                       # Camada linear final
)

# Classe Recipro-CAM
class ReciproCam:
    def __init__(self, feature_net, head_net, device):
        self.feature_net = feature_net.eval()
        self.head_net = head_net.eval()
        self.device = device
        self.softmax = torch.nn.Softmax(dim=1)
        self.gaussian = torch.tensor(
            [
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
                [1 / 8.0, 1 / 4.0, 1 / 8.0],
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
            ]
        ).to(device)

    def _mosaic_feature(self, feature_map, is_gaussian=False):
        _, num_channel, height, width = feature_map.shape
        new_features = torch.zeros(height * width, num_channel, height, width).to(
            self.device
        )
        if not is_gaussian:
            for k in range(height * width):
                for i in range(height):
                    for j in range(width):
                        if k == i * width + j:
                            new_features[k, :, i, j] = feature_map[0, :, i, j]
        else:
            for k in range(height * width):
                for i in range(height):
                    kx_s, kx_e = max(i - 1, 0), min(i + 1, height - 1)
                    sx_s = 1 if i == 0 else 0
                    sx_e = 1 if i == height - 1 else 2
                    for j in range(width):
                        ky_s, ky_e = max(j - 1, 0), min(j + 1, width - 1)
                        sy_s = 1 if j == 0 else 0
                        sy_e = 1 if j == width - 1 else 2
                        if k == i * width + j:
                            r_feature_map = (
                                feature_map[0, :, i, j]
                                .reshape(num_channel, 1, 1)
                                .repeat(1, self.gaussian.shape[0], self.gaussian.shape[1])
                            )
                            score_map = r_feature_map * self.gaussian.repeat(
                                num_channel, 1, 1
                            )
                            new_features[
                                k, :, kx_s : kx_e + 1, ky_s : ky_e + 1
                            ] = score_map[:, sx_s : sx_e + 1, sy_s : sy_e + 1]

        return new_features

    def _weight_accum(self, mosaic_predict, class_index, height, width):
        cam = torch.zeros(height, width).to(self.device)
        for i in range(height):
            for j in range(width):
                cam[i, j] = mosaic_predict[i * width + j][class_index]
        return cam

    def __call__(self, image, class_index=None):
        feature_map = self.feature_net(image)
        prediction = self.head_net(feature_map)

        if class_index is None:
            class_index = torch.argmax(prediction, dim=1).item()

        _, _, height, width = feature_map.shape

        # Gerar mapas de característica espacial mascarados
        spatial_masked_feature_map = self._mosaic_feature(
            feature_map, is_gaussian=False
        )

        # Cálculo do logit recíproco
        reciprocal_predictions = self.head_net(spatial_masked_feature_map)
        reciprocal_logits = self.softmax(reciprocal_predictions)

        # Geração do CAM
        cam = self._weight_accum(reciprocal_logits, class_index, height, width)

        # Normalização
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_index


# Inicializar Recipro-CAM
recipro_cam = ReciproCam(feature_net, head_net, device)

# Visualização do CAM
def visualize_cam(image, cam, predicted_class_idx, image_name):
    # Converter o CAM para NumPy antes de redimensionar
    cam_np = cam.detach().cpu().numpy()

    # Redimensionar o CAM para as dimensões da imagem original
    cam_resized = cv2.resize(cam_np, (image.shape[2], image.shape[1]))
    cam_resized = cam_resized - cam_resized.min()
    cam_resized = cam_resized / cam_resized.max() * 255
    cam_resized = cam_resized.astype(np.uint8)

    # Gerar o mapa de calor
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Converter a imagem para NumPy para sobreposição
    image_rgb = np.transpose(image.numpy(), (1, 2, 0))
    cam_overlay = 0.5 * image_rgb + 0.5 * heatmap

    # Exibir a sobreposição
    plt.imshow(cam_overlay)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[predicted_class_idx]} - Image: {image_name}")
    plt.show()

# Testar Recipro-CAM
for images, labels in test_dataloader_simple:
    image_path = test_dataloader_simple.dataset.imgs[0][0]
    image_name = Path(image_path).name
    images, labels = images.to(device), labels.to(device)
    cam, predicted_class_idx = recipro_cam(images, class_index=labels.item())
    visualize_cam(images[0].cpu(), cam, predicted_class_idx, image_name)
    break
