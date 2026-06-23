import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_saliency_map(model, input_tensor, device='cpu'):
    with torch.enable_grad():
        model.eval()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_()
        
        output = model(input_tensor)
        
        if output.dim() > 1 and output.shape[1] > 1:
            score = output[0, torch.argmax(output[0])]
        else:
            score = output[0, 0] if output.dim() > 1 else output[0]
            
        model.zero_grad()
        score.backward()
        
        saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
        if saliency.ndim == 3:
            saliency = np.max(saliency, axis=0)
            
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency

def generate_ensemble_saliency(models_dict, input_tensor, original_image, device='cpu', ensemble_weights=None):
    if ensemble_weights is None:
        ensemble_weights = {name: 1.0 / len(models_dict) for name in models_dict.keys()}
        
    weight_mapping = {
        'densenet121': 'densenet121',
        'resnet50': 'resnet50',
        'efficientnet': 'efficientnet',
        'tf_efficientnetv2_s': 'efficientnet'
    }
    
    ensemble_saliency = None
    total_weight = 0
    
    for model_name, model in models_dict.items():
        saliency = generate_saliency_map(model, input_tensor, device)
        
        weight_key = weight_mapping.get(model_name, model_name)
        weight = ensemble_weights.get(weight_key, 1.0 / len(models_dict))
        
        if ensemble_saliency is None:
            ensemble_saliency = saliency * weight
        else:
            ensemble_saliency += saliency * weight
            
        total_weight += weight
        
    if total_weight > 0:
        ensemble_saliency = ensemble_saliency / total_weight
        
    if ensemble_saliency.max() > 0:
        ensemble_saliency = ensemble_saliency / ensemble_saliency.max()
        
    return ensemble_saliency

def create_posthoc_visualization(original_image, saliency, alpha=0.5):
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
    h, w = original_image.shape[:2]
    saliency_resized = cv2.resize(saliency, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
        
    superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    return heatmap, superimposed

def generate_posthoc_grid(models_dict, input_tensor, original_image, device='cpu', ensemble_weights=None):
    ensemble_saliency = generate_ensemble_saliency(models_dict, input_tensor, original_image, device, ensemble_weights)
    heatmap, superimposed = create_posthoc_visualization(original_image, ensemble_saliency, alpha=0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if len(original_image.shape) == 2:
        img_display = original_image
        cmap = 'gray'
    else:
        img_display = original_image
        cmap = None
        
    axes[0].imshow(img_display, cmap=cmap)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Saliency Map', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title('Superimposed Image', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    fig.suptitle('✨ Post-Hoc Saliency Visualization - Ensemble Model', fontsize=16, fontweight='bold', y=0.98)
    
    model_names = ', '.join(models_dict.keys())
    fig.text(0.5, 0.92, f'Combined visualization from: {model_names}', ha='center', fontsize=11, style='italic', color='#666')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
