import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_heatmap(image_path):
    """
    Loads and normalizes a heatmap image.
    """
    heatmap = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if heatmap is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    
    # Convert to float and normalize to [0, 1]
    heatmap = heatmap.astype(np.float32) / 255.0
    return heatmap

def calculate_intersection_heatmap(correct_heatmap_path, incorrect_heatmap_path, output_path):
    """
    Calculates the intersection heatmap and saves it.
    """
    # Load the correct and incorrect mean heatmaps
    correct_heatmap = load_heatmap(correct_heatmap_path)
    incorrect_heatmap = load_heatmap(incorrect_heatmap_path)

    if correct_heatmap is None or incorrect_heatmap is None:
        print("Error: One or both heatmaps could not be loaded.")
        return

    # Ensure both heatmaps have the same shape
    if correct_heatmap.shape != incorrect_heatmap.shape:
        print(f"Resizing incorrect heatmap from {incorrect_heatmap.shape} to {correct_heatmap.shape}")
        incorrect_heatmap = cv2.resize(incorrect_heatmap, (correct_heatmap.shape[1], correct_heatmap.shape[0]))

    # Intersection calculation (element-wise minimum)
    intersection_heatmap = np.minimum(correct_heatmap, incorrect_heatmap)

    # Normalize the intersection map to [0, 1] for visualization
    inter_min, inter_max = np.min(intersection_heatmap), np.max(intersection_heatmap)
    intersection_heatmap = (intersection_heatmap - inter_min) / (inter_max - inter_min + 1e-8)

    # Convert to uint8 and apply JET colormap
    intersection_heatmap_uint8 = np.uint8(255 * intersection_heatmap)
    colored_intersection_heatmap = cv2.applyColorMap(intersection_heatmap_uint8, cv2.COLORMAP_JET)

    # Save the intersection heatmap
    cv2.imwrite(output_path, colored_intersection_heatmap)
    print(f"Intersection heatmap saved at: {output_path}")

    # Plot the intersection heatmap
    #plt.figure(figsize=(6, 6))
    #plt.title("Intersection Heatmap (Common Activations)")
    #plt.imshow(colored_intersection_heatmap[..., ::-1])  # Convert BGR to RGB for displaying
    #plt.axis('off')
    #plt.show()

# Example usage:
results_folder = os.path.normpath('C:/Users/anale/OneDrive/Documentos/Universidade/TESE/RESULTS/224x224_MEDVIT')

correct_heatmap_path = os.path.join(results_folder, "CORRECT MEAN MEAN HEATMAPS", "MEDVIT_cv123_FEMUR_cesarean_test_correct_mean_heatmap.jpg")
incorrect_heatmap_path = os.path.join(results_folder, "INCORRECT MEAN MEAN HEATMAPS", "MEDVIT_cv123_FEMUR_cesarean_test_incorrect_mean_heatmap.jpg")
output_path = os.path.join(results_folder, "INTERSECTION HEATMAPS", "MEDVIT_intersection_heatmap_FEMUR_cesarean_test.jpg")

calculate_intersection_heatmap(correct_heatmap_path, incorrect_heatmap_path, output_path)
