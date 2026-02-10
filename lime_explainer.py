"""
LIME Explainability for Medical Image Analysis
Fast implementation using official LIME library
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tqdm import tqdm


class FastLIMEExplainer:
    """Fast LIME explainer using official library"""
    
    def __init__(self, model, device, transform, model_name='model'):
        self.model = model
        self.device = device
        self.transform = transform
        self.model_name = model_name
        self.model.eval()
    
    def predict_fn(self, images):
        """
        Batch prediction function for LIME
        Args:
            images: Numpy array of images [N, H, W, 3] in [0, 1]
        Returns:
            predictions: Array of probabilities [N, 2]
        """
        from PIL import Image
        batch_tensors = []
        
        for img in images:
            # Convert to uint8 if needed
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img.astype(np.uint8)
            
            pil_img = Image.fromarray(img_uint8)
            tensor = self.transform(pil_img)
            batch_tensors.append(tensor)
        
        # Process in batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Return as 2-class probability
        return np.column_stack([1 - probs, probs])
    
    def explain(self, image, num_samples=50, num_features=10):
        """
        Generate LIME explanation
        Args:
            image: RGB image [H, W, 3] in [0, 255]
            num_samples: Number of perturbation samples
            num_features: Number of superpixels
        Returns:
            explanation: LIME explanation object
        """
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32)
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image_norm,
            self.predict_fn,
            top_labels=1,
            hide_color=0.5,
            num_samples=num_samples,
            batch_size=10
        )
        
        return explanation


def generate_lime_grid(models, input_tensor, original_image, device,
                       ensemble_weights=None, n_samples=50, n_segments=10):
    """
    Generate LIME grid visualization (fast version)
    Args:
        models: Dictionary of models
        input_tensor: Preprocessed input
        original_image: Original numpy image
        device: Torch device
        ensemble_weights: Model weights dict
        n_samples: Number of perturbations (default 50)
        n_segments: Number of superpixels (default 10)
    Returns:
        fig: Matplotlib figure
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare image
    if len(original_image.shape) == 2:
        rgb_image = np.stack([original_image] * 3, axis=-1)
    else:
        rgb_image = original_image
    
    rgb_image_resized = cv2.resize(rgb_image, (224, 224))
    
    if rgb_image_resized.max() > 1.0:
        rgb_image_resized = rgb_image_resized.astype(np.uint8)
    else:
        rgb_image_resized = (rgb_image_resized * 255).astype(np.uint8)
    
    model_names = {
        'densenet121': 'DenseNet121',
        'resnet50': 'ResNet50',
        'efficientnet': 'EfficientNetV2'
    }
    
    n_models = len(models)
    
    # Create figure
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4 * n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n🔍 Generating LIME with {n_samples} samples, {n_segments} superpixels...")
    
    for idx, (model_key, model) in enumerate(models.items()):
        model_display_name = model_names.get(model_key, model_key)
        print(f"\n📊 [{idx+1}/{n_models}] Processing {model_display_name}...")
        
        try:
            # Create explainer
            explainer = FastLIMEExplainer(model, device, transform, model_key)
            
            # Generate explanation (this is the slow part)
            print(f"   Generating LIME explanation...")
            explanation = explainer.explain(
                rgb_image_resized,
                num_samples=n_samples,
                num_features=n_segments
            )
            
            # Get prediction
            pred_probs = explainer.predict_fn(rgb_image_resized[np.newaxis, :] / 255.0)
            prediction = pred_probs[0][1]  # Probability of positive class
            
            # Get explanation visualization
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=n_segments,
                hide_rest=False
            )
            
            # Create boundary visualization
            img_norm = rgb_image_resized.astype(np.float32) / 255.0
            boundary_img = mark_boundaries(img_norm, mask, color=(0, 1, 0), mode='thick')
            
            # Create heatmap
            heatmap = np.zeros(rgb_image_resized.shape[:2], dtype=np.float32)
            for segment_id in np.unique(mask):
                if segment_id > 0:
                    heatmap[mask == segment_id] = 1.0
            
            # Apply colormap
            heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
            heatmap_img = 0.5 * heatmap_colored + 0.5 * img_norm
            
            # Get model weight
            model_weight = ensemble_weights.get(model_key, 1.0) if ensemble_weights else 1.0
            
            # Column 1: Boundaries
            axes[idx, 0].imshow(boundary_img)
            axes[idx, 0].set_title(
                f'{model_display_name}\nProb: {prediction:.3f} | Weight: {model_weight:.2f}',
                fontsize=11, fontweight='bold'
            )
            axes[idx, 0].axis('off')
            
            # Column 2: Heatmap
            axes[idx, 1].imshow(heatmap_img)
            axes[idx, 1].set_title('Importance Heatmap', fontsize=10)
            axes[idx, 1].axis('off')
            
            # Column 3: Original
            axes[idx, 2].imshow(rgb_image_resized)
            axes[idx, 2].set_title('Original Image', fontsize=10)
            axes[idx, 2].axis('off')
            
            print(f"   ✅ {model_display_name} complete!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            for col in range(3):
                axes[idx, col].text(
                    0.5, 0.5, f'Error:\n{str(e)[:40]}',
                    ha='center', va='center', fontsize=9, color='red'
                )
                axes[idx, col].axis('off')
    
    # Add main title
    fig.suptitle('LIME Explainability - Model Interpretability Analysis',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(f"\n✅ LIME visualization complete!")
    
    return fig


def generate_lime_comparison(models, input_tensor, original_image, device,
                             ensemble_weights=None, n_samples=50, n_segments=10):
    """
    Generate comparison view (fast version)
    Args:
        n_samples: Number of perturbations (default 50)
        n_segments: Number of superpixels (default 10)
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare image
    if len(original_image.shape) == 2:
        rgb_image = np.stack([original_image] * 3, axis=-1)
    else:
        rgb_image = original_image
    
    rgb_image_resized = cv2.resize(rgb_image, (224, 224))
    
    if rgb_image_resized.max() > 1.0:
        rgb_image_resized = rgb_image_resized.astype(np.uint8)
    else:
        rgb_image_resized = (rgb_image_resized * 255).astype(np.uint8)
    
    model_names = {
        'densenet121': 'DenseNet121',
        'resnet50': 'ResNet50',
        'efficientnet': 'EfficientNetV2'
    }
    
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(8, 4 * n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n🔍 Generating LIME comparison with {n_samples} samples...")
    
    for idx, (model_key, model) in enumerate(models.items()):
        model_display_name = model_names.get(model_key, model_key)
        print(f"\n📊 [{idx+1}/{n_models}] Processing {model_display_name}...")
        
        try:
            explainer = FastLIMEExplainer(model, device, transform, model_key)
            
            print(f"   Generating LIME explanation...")
            explanation = explainer.explain(
                rgb_image_resized,
                num_samples=n_samples,
                num_features=n_segments
            )
            
            pred_probs = explainer.predict_fn(rgb_image_resized[np.newaxis, :] / 255.0)
            prediction = pred_probs[0][1]
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=n_segments,
                hide_rest=False
            )
            
            img_norm = rgb_image_resized.astype(np.float32) / 255.0
            boundary_img = mark_boundaries(img_norm, mask, color=(0, 1, 0), mode='thick')
            
            heatmap = np.zeros(rgb_image_resized.shape[:2], dtype=np.float32)
            for segment_id in np.unique(mask):
                if segment_id > 0:
                    heatmap[mask == segment_id] = 1.0
            
            heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
            heatmap_img = 0.5 * heatmap_colored + 0.5 * img_norm
            
            model_weight = ensemble_weights.get(model_key, 1.0) if ensemble_weights else 1.0
            
            # Column 1: Boundaries
            axes[idx, 0].imshow(boundary_img)
            axes[idx, 0].set_title(
                f'{model_display_name}\nProb: {prediction:.3f}',
                fontsize=11, fontweight='bold'
            )
            axes[idx, 0].axis('off')
            
            # Column 2: Heatmap
            axes[idx, 1].imshow(heatmap_img)
            axes[idx, 1].set_title('Heatmap', fontsize=10)
            axes[idx, 1].axis('off')
            
            print(f"   ✅ Complete!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            for col in range(2):
                axes[idx, col].text(
                    0.5, 0.5, f'Error:\n{str(e)[:50]}',
                    ha='center', va='center', fontsize=9, color='red'
                )
                axes[idx, col].axis('off')
    
    fig.suptitle('LIME Explainability Comparison',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print(f"\n✅ LIME comparison complete!")
    
    return fig
