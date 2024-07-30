import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import requests

# Load a pre-trained model
model = models.resnet101(weights='DEFAULT')
model.eval()

# Load ImageNet class labels
def load_imagenet_labels(url='https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'):
    response = requests.get(url)
    return response.json()

imagenet_labels = load_imagenet_labels()

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = "/content/dog1.jpg"
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)

# Function to compute FisherCAM
def generate_fisher_cam(feature_map, fisher_weights):
    size_upsample = (224, 224)
    nc, h, w = feature_map.shape
    cam = fisher_weights.dot(feature_map.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, size_upsample)

def fisher_cam(model, image, target_class):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = model.layer4[-1].register_forward_hook(forward_hook)
    handle_bw = model.layer4[-1].register_backward_hook(backward_hook)

    output = model(image)
    model.zero_grad()

    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    handle_fw.remove()
    handle_bw.remove()

    grads_val = gradients[0].cpu().data.numpy()[0]
    target = features[0].cpu().data.numpy()[0]

    squared_grads = np.square(grads_val)
    fisher_weights = np.mean(squared_grads, axis=(1, 2))

    cam = generate_fisher_cam(target, fisher_weights)

    return cam, grads_val

def mixcam_attack_untargeted(model, image, epsilon, num_iterations, decay_factor, q_percentile, fusion_ratio):
    alpha = epsilon / num_iterations
    perturbed_image = image.clone()
    momentum = torch.zeros_like(image)
    gradients_list = []

    for i in range(num_iterations):
        perturbed_image.requires_grad_()

        output = model(perturbed_image)
        pred_class = output.argmax().item()

        cam, gradient = fisher_cam(model, perturbed_image, pred_class)
        gradients_list.append(gradient)

        mask = cam <= np.percentile(cam, q_percentile)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)

        #threshold = np.percentile(cam, q_percentile)
        #relu_cam = torch.relu(torch.tensor(cam - threshold, dtype=torch.float32)).unsqueeze(0).unsqueeze(0).to(image.device)
        #masked_image = image * relu_cam

        masked_image = image * mask
        mixed_image = fusion_ratio * image + (1 - fusion_ratio) * masked_image
        mixed_image = Variable(mixed_image, requires_grad=True)

        output = model(mixed_image)
        loss = F.cross_entropy(output, torch.LongTensor([pred_class]).to(image.device))
        model.zero_grad()
        loss.backward()

        gradient = mixed_image.grad.data
        momentum = decay_factor * momentum + gradient / torch.norm(gradient, p=1)
        #momentum = gradient
        perturbed_image = perturbed_image + alpha * torch.sign(momentum)

        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()

    return perturbed_image, pred_class, cam, mask, masked_image, mixed_image, gradients_list

# Parameters
epsilon = 0.05
num_iterations = 10
decay_factor = 1.0
q_percentile = 80
fusion_ratio = 0.6

# Generate adversarial example
perturbed_image, pred_class, cam, mask, masked_image, mixed_image, gradients_list = mixcam_attack_untargeted(
    model, image_tensor, epsilon, num_iterations, decay_factor, q_percentile, fusion_ratio
)

# Convert tensors to numpy arrays for visualization
def tensor_to_np(tensor):
    return tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

original_image_np = tensor_to_np(image_tensor)
perturbed_image_np = tensor_to_np(perturbed_image)
masked_image_np = tensor_to_np(masked_image)
mixed_image_np = tensor_to_np(mixed_image)

# Compute alpha * torch.sign(momentum)
def compute_alpha_sign_momentum(alpha, momentum_tensor):
    return  alpha, momentum_tensor, alpha * torch.sign(momentum_tensor).squeeze().detach().cpu().numpy().transpose(1, 2, 0)

# Use the final computed momentum tensor for visualization
momentum_tensor = torch.zeros_like(image_tensor)
alpha, momentum_tensor,momentum_image = compute_alpha_sign_momentum(epsilon / num_iterations, momentum_tensor)

# Overlay FisherCAM heatmap on the original image
def overlay_heatmap_on_image(heatmap, image, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255.0
    image_resized = np.uint8(image * 255)  # Convert image to [0, 255] range
    image_resized = np.float32(image_resized) / 255.0
    overlaid_image = np.clip(heatmap_colored * alpha + image_resized, 0, 1)
    return overlaid_image

overlaid_image = overlay_heatmap_on_image(cam, original_image_np)

# Show the images and matrices
fig, ax = plt.subplots(2, 4, figsize=(25, 15))

# Original Image
ax[0, 0].imshow(original_image_np)
ax[0, 0].set_title("Original Image")
ax[0, 0].axis('off')

# FisherCAM Heatmap
ax[0, 1].imshow(cam, cmap='jet')
ax[0, 1].set_title("FisherCAM Heatmap")
ax[0, 1].axis('off')

# Overlay FisherCAM on original image
ax[0, 2].imshow(overlaid_image)
ax[0, 2].set_title("Overlay Heatmap")
ax[0, 2].axis('off')

# Mask
ax[0, 3].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
ax[0, 3].set_title("Mask")
ax[0, 3].axis('off')

# Masked Image
ax[1, 0].imshow(masked_image_np)
ax[1, 0].set_title("Masked Image")
ax[1, 0].axis('off')

# Mixed Image
ax[1, 1].imshow(mixed_image_np)
ax[1, 1].set_title("Mixed Image")
ax[1, 1].axis('off')

ax[1, 2].imshow(momentum_image)
ax[1, 2].set_title("Alpha * Sign(Momentum)")
ax[1, 2].axis('off') 

# Adversarial Image
ax[1, 3].imshow(perturbed_image_np)
ax[1, 3].set_title(f"Adversarial Image\nPredicted class: {pred_class}")
ax[1, 3].axis('off')


# Print labels

print(f"Adversarial Image Label: {imagenet_labels[pred_class]}")

plt.show()

# Save adversarial image
save_image(perturbed_image, 'adversarial_image.png')
