import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Load and preprocess the image
def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Display image
def display_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

img_path = '/path/image.jpg'
img_array = load_image(img_path)
display_image(img_path)

# Load VGG16 model
model = VGG16(weights='imagenet')

# Set the class index to robin (as per your example)
class_idx = 222
class_label = decode_predictions(np.eye(1, 1000, class_idx))[0][0][1]
print(f'Using class index: {class_idx}')
print(f'Class label: {class_label}')

# Define the model for Grad-CAM
last_conv_layer = model.get_layer('block5_conv3')
heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])

# Function to compute Grad-CAM heatmap
def compute_grad_cam(model, img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_array)
        output = predictions[:, class_idx]
    grads = tape.gradient(output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Function to compute FisherCAM heatmap
def fisher_cam_heatmap(model, img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_array)
        output = predictions[:, class_idx]
    grads = tape.gradient(output, conv_output)
    squared_grads = tf.square(grads)  # Square the gradients

    conv_output = conv_output[0].numpy()
    squared_grads = squared_grads[0].numpy()

    novelcam = np.mean(squared_grads * conv_output, axis=-1)
    novelcam = np.maximum(novelcam, 0)
    novelcam /= np.max(novelcam)
    return novelcam

# Compute heatmaps
gradcam = compute_grad_cam(model, img_array, class_idx)
fisher_heatmap = fisher_cam_heatmap(model, img_array, class_idx)

# Display heatmaps without overlay
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(gradcam, cmap='viridis')
axes[0].set_title('Grad-CAM')
axes[0].axis('off')

axes[1].imshow(square_fisher_heatmap, cmap='viridis')
axes[1].set_title('FisherCAM')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Overlay heatmaps
def overlay_heatmap(img_path, heatmap, title, alpha=0.6, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

overlay_heatmap(img_path, gradcam, f'Grad-CAM: {class_label} (class {class_idx})')
overlay_heatmap(img_path, square_fisher_heatmap, f'FisherCAM: {class_label} (class {class_idx})')

# Plot combined 3D histograms of heatmap values
def plot_combined_3d_histograms(grad_cam_map, square_fisher_cam_map):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    bins = 50
    hist1, bin_edges = np.histogram(grad_cam_map.flatten(), bins=bins)
    hist2, _ = np.histogram(square_fisher_cam_map.flatten(), bins=bins)

    xpos = (bin_edges[:-1] + bin_edges[1:]) / 2
    ypos1 = np.zeros_like(xpos)
    ypos2 = np.ones_like(xpos)
    zpos = np.zeros_like(xpos)

    dx = dy = 0.5 * np.ones_like(zpos)

    ax.bar3d(xpos, ypos1, zpos, dx, dy, hist1, color='r', alpha=0.7)
    ax.bar3d(xpos, ypos2, zpos, dx, dy, hist2, color='b', alpha=0.7)

    ax.set_xlabel('Heatmap Value')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Grad-CAM', 'FisherCAM'])
    ax.set_title('Combined 3D Histograms of Heatmap Values')

    # Create custom legend
    custom_lines = [Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='b', lw=4)]
    ax.legend(custom_lines, ['Grad-CAM', 'FisherCAM'])

    plt.show()

plot_combined_3d_histograms(gradcam, square_fisher_heatmap)
