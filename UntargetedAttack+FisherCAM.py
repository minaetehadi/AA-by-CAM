# Untargeted attack using FisherCAM

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

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("/content/Ax (2).jpg")
image = transform(image).unsqueeze(0)

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

# Compute gradients and generate FisherCAM
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

    return cam

# MixCAM attack generation (untargeted)
def mixcam_attack_untargeted(model, image, epsilon, num_iterations, decay_factor, q_percentile, fusion_ratio):
    alpha = epsilon / num_iterations
    perturbed_image = image.clone()
    momentum = torch.zeros_like(image)

    for i in range(num_iterations):
        perturbed_image.requires_grad_()

        output = model(perturbed_image)
        pred_class = output.argmax().item()

        cam = fisher_cam(model, perturbed_image, pred_class)
        mask = cam >= np.percentile(cam, q_percentile)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(image.device)

        masked_image = image * mask
        mixed_image = fusion_ratio * image + (1 - fusion_ratio) * masked_image
        mixed_image = Variable(mixed_image, requires_grad=True)

        output = model(mixed_image)
        loss = F.cross_entropy(output, torch.LongTensor([pred_class]).to(image.device))
        model.zero_grad()
        loss.backward()

        gradient = mixed_image.grad.data
        momentum = decay_factor * momentum + gradient / torch.norm(gradient, p=1)
        perturbed_image = perturbed_image + alpha * torch.sign(momentum)

        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()

    return perturbed_image, pred_class

# Parameters
epsilon = 0.03
num_iterations = 10
decay_factor = 1.0
q_percentile = 95
fusion_ratio = 0.5

# Generate adversarial example
perturbed_image, pred_class = mixcam_attack_untargeted(model, image, epsilon, num_iterations, decay_factor, q_percentile, fusion_ratio)

# Show the original and adversarial image
original_image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
perturbed_image_np = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(perturbed_image_np)
ax[1].set_title(f"Adversarial Image\nPredicted class: {pred_class}")
ax[1].axis('off')

plt.show()

# Save adversarial image
save_image(perturbed_image, 'adversarial_image.png')
