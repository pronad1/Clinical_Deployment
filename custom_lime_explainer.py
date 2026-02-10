"""
Custom LIME-like Explainability (No External Dependencies)
Uses only numpy, opencv, torch, and matplotlib
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import Ridge


class CustomLIMEExplainer:
    """Lightweight LIME-like explainer using only standard dependencies"""
    
    def __init__(self, model, device, model_name='model'):
        """
        Initialize explainer
        Args:
            model: PyTorch model
            device: torch device
            model_name: Name for display
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.model.eval()
    
    def segment_image_slic(self, image, n_segments=50):
        """
        Segment image using SLIC superpixels (OpenCV)
        Args:
            image: RGB image [H, W, 3]
            n_segments: Number of segments
        Returns:
            segments: Segment labels [H, W]
        """
        try:
            # Try using OpenCV's SLIC if available
            slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=int(np.sqrt(image.size / n_segments)))
            slic.iterate(10)
            segments = slic.getLabels()
            return segments
        except:
            # Fallback to simple grid-based segmentation
            return self.segment_image_grid(image, n_segments)
    
    def segment_image_grid(self, image, n_segments=50):
        """
        Fallback: Grid-based segmentation
        Args:
            image: RGB image [H, W, 3]
            n_segments: Approximate number of segments
        Returns:
            segments: Segment labels [H, W]
        """
        height, width = image.shape[:2]
        n_rows = int(np.sqrt(n_segments * height / width))
        n_cols = int(np.sqrt(n_segments * width / height))
        
        segments = np.zeros((height, width), dtype=np.int32)
        row_size = height // n_rows
        col_size = width // n_cols
        
        segment_id = 0
        for i in range(n_rows):
            for j in range(n_cols):
                r_start = i * row_size
                r_end = (i + 1) * row_size if i < n_rows - 1 else height
                c_start = j * col_size
                c_end = (j + 1) * col_size if j < n_cols - 1 else width
                segments[r_start:r_end, c_start:c_end] = segment_id
                segment_id += 1
        
        return segments
    
    def predict_fn(self, images, transform):
        """
        Batch prediction function
        Args:
            images: List of RGB images
            transform: Torchvision transform
        Returns:
            predictions: Array of probabilities
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for img in images:
                from PIL import Image
                pil_img = Image.fromarray(img)
                img_tensor = transform(pil_img).unsqueeze(0).to(self.device)
                output = self.model(img_tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        
        return np.array(predictions)
    
    def generate_perturbations(self, image, segments, n_samples=500):
        """
        Generate perturbed versions of the image
        Args:
            image: Original image [H, W, 3]
            segments: Segment labels [H, W]
            n_samples: Number of samples to generate
        Returns:
            perturbed_images: List of perturbed images
            segment_masks: Binary masks indicating which segments are active
        """
        n_segments = len(np.unique(segments))
        perturbed_images = []
        segment_masks = []
        
        # Always include original image (all segments on)
        perturbed_images.append(image.copy())
        segment_masks.append(np.ones(n_segments))
        
        # Generate random perturbations
        for _ in range(n_samples - 1):
            # Randomly decide which segments to keep (30-70% on average)
            mask = np.random.randint(0, 2, n_segments)
            
            # Create perturbed image
            perturbed = image.copy()
            for seg_id in range(n_segments):
                if mask[seg_id] == 0:
                    # Mask out this segment (set to mean gray)
                    perturbed[segments == seg_id] = [128, 128, 128]
            
            perturbed_images.append(perturbed)
            segment_masks.append(mask)
        
        return perturbed_images, np.array(segment_masks)
    
    def explain(self, image, transform, n_segments=50, n_samples=500):
        """
        Generate explanation for an image
        Args:
            image: RGB image [H, W, 3] in range [0, 255]
            transform: Torchvision transform
            n_segments: Number of superpixels
            n_samples: Number of perturbations
        Returns:
            segments: Segment labels
            weights: Importance weights for each segment
            prediction: Model's prediction
        """
        # Ensure correct format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Segment the image
        segments = self.segment_image_slic(image, n_segments)
        
        # Generate perturbations
        perturbed_images, segment_masks = self.generate_perturbations(
            image, segments, n_samples
        )
        
        # Get predictions for all perturbations
        predictions = self.predict_fn(perturbed_images, transform)
        
        # Fit linear model (Ridge regression)
        model = Ridge(alpha=1.0)
        model.fit(segment_masks, predictions)
        
        # Get weights (importance of each segment)
        weights = model.coef_
        
        # Get original prediction
        original_pred = predictions[0]
        
        return segments, weights, original_pred
    
    def create_visualization(self, image, segments, weights, num_features=10):
        """
        Create boundary and heatmap visualizations
        Args:
            image: Original image
            segments: Segment labels
            weights: Importance weights
            num_features: Number of top features to show
        Returns:
            boundary_img: Image with boundaries
            heatmap_img: Heatmap overlay
            mask: Binary mask of important regions
        """
        # Get top segments
        top_indices = np.argsort(np.abs(weights))[-num_features:]
        
        # Create binary mask
        mask = np.zeros(segments.shape, dtype=np.uint8)
        for idx in top_indices:
            if weights[idx] > 0:  # Only positive contributions
                mask[segments == idx] = 1
        
        # Create boundary image
        boundary_img = image.copy() / 255.0
        
        # Find boundaries
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_img_with_contours = cv2.cvtColor((boundary_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.drawContours(boundary_img_with_contours, contours, -1, (0, 255, 0), 2)
        boundary_img = cv2.cvtColor(boundary_img_with_contours, cv2.COLOR_BGR2RGB) / 255.0
        
        # Also draw segment boundaries for all top segments
        for idx in top_indices:
            seg_mask = (segments == idx).astype(np.uint8)
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_img_bgr = cv2.cvtColor((boundary_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.drawContours(boundary_img_bgr, contours, -1, (0, 255, 0), 1)
            boundary_img = cv2.cvtColor(boundary_img_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        # Create heatmap
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for idx in range(len(weights)):
            if weights[idx] > 0:
                heatmap[segments == idx] = weights[idx]
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        
        # Blend with original image
        alpha = 0.6
        img_normalized = image / 255.0 if image.max() > 1 else image
        heatmap_img = alpha * heatmap_colored + (1 - alpha) * img_normalized
        
        return boundary_img, heatmap_img, mask


def generate_custom_lime_grid(models, input_tensor, original_image, device,
                               ensemble_weights=None, n_samples=300, n_segments=50):
    """
    Generate LIME-like explanations grid for multiple models
    Args:
        models: Dictionary of models
        input_tensor: Preprocessed tensor
        original_image: Original image array
        device: PyTorch device
        ensemble_weights: Model weights
        n_samples: Number of perturbation samples
        n_segments: Number of superpixels
    Returns:
        fig: Matplotlib figure
    """
    from torchvision import transforms
    
    # Transform for predictions
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
    
    # Resize to 224x224
    rgb_image_resized = cv2.resize(rgb_image, (224, 224))
    
    if rgb_image_resized.max() <= 1.0:
        rgb_image_resized = (rgb_image_resized * 255).astype(np.uint8)
    else:
        rgb_image_resized = rgb_image_resized.astype(np.uint8)
    
    model_display_names = {
        'densenet121': 'DenseNet121',
        'resnet50': 'ResNet50',
        'efficientnet': 'EfficientNetV2'
    }
    
    # Create figure
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate explanations for each model
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"Generating custom LIME explanation for {model_name}...")
        
        try:
            # Create explainer
            explainer = CustomLIMEExplainer(model, device, model_name)
            
            # Generate explanation
            segments, weights, prediction = explainer.explain(
                rgb_image_resized,
                transform,
                n_segments=n_segments,
                n_samples=n_samples
            )
            
            # Create visualizations
            boundary_img, heatmap_img, _ = explainer.create_visualization(
                rgb_image_resized,
                segments,
                weights,
                num_features=10
            )
            
            weight_str = f" (w={ensemble_weights.get(model_name, 1.0):.2f})" if ensemble_weights else ""
            
            # Row 1: Boundaries
            axes[0, idx].imshow(boundary_img)
            axes[0, idx].set_title(
                f'{model_display_names.get(model_name, model_name)}\nProb: {prediction:.3f}{weight_str}',
                fontsize=10, fontweight='bold'
            )
            axes[0, idx].axis('off')
            
            # Row 2: Heatmap
            axes[1, idx].imshow(heatmap_img)
            axes[1, idx].set_title('Importance Heatmap', fontsize=9)
            axes[1, idx].axis('off')
            
        except Exception as e:
            print(f"Error generating explanation for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            axes[0, idx].text(0.5, 0.5, f'Error:\n{str(e)[:50]}',
                            ha='center', va='center', fontsize=8, color='red')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
    
    fig.suptitle('LIME-like Explainability Analysis',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def generate_custom_lime_comparison(models, input_tensor, original_image, device,
                                     ensemble_weights=None, n_samples=300, n_segments=50):
    """
    Generate detailed comparison view
    Args:
        models: Dictionary of models
        input_tensor: Preprocessed tensor
        original_image: Original image
        device: PyTorch device
        ensemble_weights: Model weights
        n_samples: Number of samples
        n_segments: Number of segments
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
    
    if rgb_image_resized.max() <= 1.0:
        rgb_image_resized = (rgb_image_resized * 255).astype(np.uint8)
    else:
        rgb_image_resized = rgb_image_resized.astype(np.uint8)
    
    model_display_names = {
        'densenet121': 'DenseNet121',
        'resnet50': 'ResNet50',
        'efficientnet': 'EfficientNetV2'
    }
    
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4 * n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"Generating comparison for {model_name}...")
        
        try:
            explainer = CustomLIMEExplainer(model, device, model_name)
            
            segments, weights, prediction = explainer.explain(
                rgb_image_resized,
                transform,
                n_segments=n_segments,
                n_samples=n_samples
            )
            
            boundary_img, heatmap_img, _ = explainer.create_visualization(
                rgb_image_resized,
                segments,
                weights,
                num_features=10
            )
            
            # Column 1: Original
            axes[idx, 0].imshow(rgb_image_resized)
            axes[idx, 0].set_title(f'{model_display_names.get(model_name, model_name)} - Original',
                                 fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Column 2: Boundaries
            axes[idx, 1].imshow(boundary_img)
            axes[idx, 1].set_title(f'Explanation (Prob: {prediction:.3f})', fontsize=11)
            axes[idx, 1].axis('off')
            
            # Column 3: Heatmap
            axes[idx, 2].imshow(heatmap_img)
            axes[idx, 2].set_title('Importance Heatmap', fontsize=11)
            axes[idx, 2].axis('off')
            
        except Exception as e:
            print(f"Error for {model_name}: {e}")
            for col in range(3):
                axes[idx, col].text(0.5, 0.5, f'Error: {str(e)[:30]}',
                                  ha='center', va='center', fontsize=9, color='red')
                axes[idx, col].axis('off')
    
    fig.suptitle('LIME-like Explainability Comparison',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig
