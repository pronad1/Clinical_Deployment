"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Generator
Generates attention heatmaps and superimposed images for model interpretability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class GradCAM:
    """Generate Grad-CAM visualizations for CNN models"""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        Args:
            model: PyTorch model
            target_layer: Target layer for visualization (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class index (None for regression/binary classification)
        Returns:
            cam: Class activation map as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # For binary classification/regression
        if target_class is None:
            score = output[0, 0] if output.dim() > 1 else output[0]
        else:
            score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().numpy()[0]  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    @staticmethod
    def get_target_layer(model, model_name):
        """
        Get the target layer for Grad-CAM based on model architecture
        Args:
            model: PyTorch model
            model_name: Name of the model architecture
        Returns:
            target_layer: The last convolutional layer
        """
        if 'densenet' in model_name.lower():
            return model.features[-1]  # Last dense block
        elif 'resnet' in model_name.lower():
            return model.layer4[-1]  # Last residual block
        elif 'efficientnet' in model_name.lower():
            return model.conv_head if hasattr(model, 'conv_head') else model.blocks[-1]
        else:
            # Generic: find last Conv2d layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
        return None


def create_gradcam_visualization(original_image, cam, alpha=0.5):
    """
    Create Grad-CAM visualization with heatmap and superimposed image
    Args:
        original_image: Original image as numpy array [H, W] or [H, W, 3]
        cam: Class activation map [H', W']
        alpha: Transparency for overlay (0-1)
    Returns:
        heatmap: Colored heatmap image
        superimposed: Original image with heatmap overlay
    """
    # Ensure original image is RGB
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Resize CAM to match original image size
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap (using JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Normalize original image to [0, 255]
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
    
    # Create superimposed image
    superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return heatmap, superimposed


def generate_ensemble_gradcam(models_dict, input_tensor, original_image, device='cpu', ensemble_weights=None):
    """
    Generate ensemble Grad-CAM by combining CAMs from multiple models
    Args:
        models_dict: Dictionary of {model_name: model}
        input_tensor: Preprocessed input tensor [1, C, H, W]
        original_image: Original image as numpy array
        device: Device to run models on
        ensemble_weights: Dictionary of {model_name: weight} for combining CAMs
    Returns:
        ensemble_cam: Combined CAM from all models
    """
    # Default ensemble weights (equal weighting)
    if ensemble_weights is None:
        ensemble_weights = {name: 1.0 / len(models_dict) for name in models_dict.keys()}
    
    # Map model names to weight keys
    weight_mapping = {
        'densenet121': 'densenet121',
        'resnet50': 'resnet50',
        'efficientnet': 'efficientnet',
        'tf_efficientnetv2_s': 'efficientnet'
    }
    
    ensemble_cam = None
    total_weight = 0
    
    for model_name, model in models_dict.items():
        # Get target layer
        target_layer = GradCAM.get_target_layer(model, model_name)
        
        if target_layer is None:
            print(f"Warning: Could not find target layer for {model_name}")
            continue
        
        # Generate Grad-CAM for this model
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor.to(device))
        
        # Get weight for this model
        weight_key = weight_mapping.get(model_name, model_name)
        weight = ensemble_weights.get(weight_key, 1.0 / len(models_dict))
        
        # Add weighted CAM to ensemble
        if ensemble_cam is None:
            ensemble_cam = cam * weight
        else:
            ensemble_cam += cam * weight
        
        total_weight += weight
        
        print(f"  ✓ Added {model_name} CAM (weight: {weight:.2f})")
    
    # Normalize by total weight
    if total_weight > 0:
        ensemble_cam = ensemble_cam / total_weight
    
    # Ensure values are in [0, 1]
    if ensemble_cam.max() > 0:
        ensemble_cam = ensemble_cam / ensemble_cam.max()
    
    return ensemble_cam


def generate_gradcam_grid(models_dict, input_tensor, original_image, device='cpu', ensemble_weights=None):
    """
    Generate Ensemble Grad-CAM visualization (single row)
    Args:
        models_dict: Dictionary of {model_name: model}
        input_tensor: Preprocessed input tensor [1, C, H, W]
        original_image: Original image as numpy array
        device: Device to run models on
        ensemble_weights: Dictionary of weights for combining models
    Returns:
        fig: Matplotlib figure with ensemble visualization
    """
    print("🔥 Generating Ensemble Grad-CAM Analysis...")
    
    # Generate ensemble CAM
    ensemble_cam = generate_ensemble_gradcam(models_dict, input_tensor, original_image, device, ensemble_weights)
    
    # Create visualizations
    heatmap, superimposed = create_gradcam_visualization(original_image, ensemble_cam, alpha=0.5)
    
    # Create single row figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare original image for display
    if len(original_image.shape) == 2:
        img_display = original_image
        cmap = 'gray'
    else:
        img_display = original_image
        cmap = None
    
    # Original image
    axes[0].imshow(img_display, cmap=cmap)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Attention Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Superimposed
    axes[2].imshow(superimposed)
    axes[2].set_title('Superimposed Image', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add title
    fig.suptitle('📊 Grad-CAM Analysis - Ensemble Model', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add subtitle with model info
    model_names = ', '.join(models_dict.keys())
    fig.text(0.5, 0.92, f'Combined visualization from: {model_names}', 
             ha='center', fontsize=11, style='italic', color='#666')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    print("✅ Ensemble Grad-CAM generated successfully!")
    return fig


def save_gradcam_visualization(models_dict, input_tensor, original_image, 
                                output_path, device='cpu', dpi=150, ensemble_weights=None):
    """
    Generate and save ensemble Grad-CAM visualization
    Args:
        models_dict: Dictionary of {model_name: model}
        input_tensor: Preprocessed input tensor
        original_image: Original image as numpy array
        output_path: Path to save the figure
        device: Device to run models on
        dpi: DPI for saved figure
        ensemble_weights: Dictionary of weights for combining models
    """
    fig = generate_gradcam_grid(models_dict, input_tensor, original_image, device, ensemble_weights)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def generate_single_gradcam(model, model_name, input_tensor, original_image, device='cpu'):
    """
    Generate Grad-CAM for a single model
    Args:
        model: PyTorch model
        model_name: Name of the model
        input_tensor: Preprocessed input tensor [1, C, H, W]
        original_image: Original image as numpy array
        device: Device to run model on
    Returns:
        Dictionary with visualization components
    """
    # Get target layer
    target_layer = GradCAM.get_target_layer(model, model_name)
    
    if target_layer is None:
        raise ValueError(f"Could not find target layer for {model_name}")
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor.to(device))
    
    # Create visualizations
    heatmap, superimposed = create_gradcam_visualization(original_image, cam)
    
    return {
        'cam': cam,
        'heatmap': heatmap,
        'superimposed': superimposed,
        'original': original_image
    }
