"""
Qualitative Segmentation Visualization Generator
Generates comparison visualizations for segmentation tasks including:
- Ground truth masks
- Predicted masks
- Error analysis
- Boundary details
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def create_binary_mask_from_detections(detections, image_shape, threshold=0.5):
    """
    Create binary segmentation mask from YOLO detection results
    Args:
        detections: YOLO detection results
        image_shape: Tuple of (height, width)
        threshold: Confidence threshold for detections
    Returns:
        mask: Binary mask [H, W]
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if detections is None or len(detections) == 0:
        return mask
    
    # Process each detection
    try:
        for detection in detections:
            # Check if detection has masks (segmentation)
            if hasattr(detection, 'masks') and detection.masks is not None:
                mask_data = detection.masks.data.cpu().numpy()
                for m in mask_data:
                    # Resize mask to image size
                    m_resized = cv2.resize(m, (w, h))
                    mask = np.maximum(mask, (m_resized > 0.5).astype(np.uint8))
            # Otherwise use bounding boxes
            elif hasattr(detection, 'boxes'):
                boxes = detection.boxes
                for box in boxes:
                    if box.conf >= threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        mask[y1:y2, x1:x2] = 1
    except Exception as e:
        print(f"Error creating mask from detections: {e}")
    
    return mask


def compute_error_mask(ground_truth, prediction):
    """
    Compute error mask showing false positives and false negatives
    Args:
        ground_truth: Binary ground truth mask [H, W]
        prediction: Binary predicted mask [H, W]
    Returns:
        error_mask: Error visualization (0=correct, 1=FN, 2=FP)
    """
    error_mask = np.zeros_like(ground_truth, dtype=np.uint8)
    
    # False Negatives: GT=1, Pred=0
    error_mask[(ground_truth == 1) & (prediction == 0)] = 1
    
    # False Positives: GT=0, Pred=1
    error_mask[(ground_truth == 0) & (prediction == 1)] = 2
    
    return error_mask


def visualize_error_analysis(original_image, error_mask):
    """
    Create error analysis visualization with color coding:
    - Green: False Negatives (missed lesions)
    - Red: False Positives (false alarms)
    Args:
        original_image: Original image [H, W] or [H, W, 3]
        error_mask: Error mask (0=correct, 1=FN, 2=FP)
    Returns:
        error_viz: RGB visualization of errors
    """
    # Create black background for better visibility
    h, w = error_mask.shape
    error_viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Check if there are any errors
    has_errors = np.any(error_mask > 0)
    
    if has_errors:
        # Use pure bright colors on black background for maximum visibility
        # False Negatives in Bright Green (missed detections)
        error_viz[error_mask == 1] = [0, 255, 0]
        
        # False Positives in Bright Red (false alarms)
        error_viz[error_mask == 2] = [255, 0, 0]
    else:
        # If no errors, show darkened version of original to indicate success
        if len(original_image.shape) == 2:
            gray_img = original_image
        else:
            gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize and darken
        if gray_img.max() > 1.0:
            gray_img = gray_img.astype(np.float32) / 255.0
        
        # Very dark background (20% brightness)
        darkened = (gray_img * 0.2 * 255).astype(np.uint8)
        error_viz = cv2.cvtColor(darkened, cv2.COLOR_GRAY2RGB)
    
    return error_viz


def visualize_boundary_details(original_image, ground_truth, prediction):
    """
    Create boundary visualization showing overlaps and differences
    Args:
        original_image: Original image [H, W] or [H, W, 3]
        ground_truth: Binary ground truth mask [H, W]
        prediction: Binary predicted mask [H, W]
    Returns:
        boundary_viz: RGB visualization with boundaries
    """
    # Start with darkened version of original for better overlay visibility
    if len(original_image.shape) == 2:
        gray_img = original_image
    else:
        gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
    # Normalize to [0, 1]
    if gray_img.max() > 1.0:
        gray_img = gray_img.astype(np.float32) / 255.0
    
    # Create darkened background (40% brightness for better overlay visibility)
    darkened = (gray_img * 0.4 * 255).astype(np.uint8)
    boundary_viz = cv2.cvtColor(darkened, cv2.COLOR_GRAY2RGB)
    
    # Find boundaries using morphological operations with thicker boundaries
    kernel = np.ones((3, 3), np.uint8)
    
    # Create filled region overlays with semi-transparency
    overlay = boundary_viz.copy()
    
    # Ground truth regions in green with higher opacity
    overlay[ground_truth > 0] = [0, 200, 0]
    
    # Prediction regions in yellow-green for overlap visibility
    overlay[prediction > 0] = [180, 255, 80]
    
    # Blend overlay (more visible: 40% background + 60% overlay)
    boundary_viz = cv2.addWeighted(boundary_viz, 0.4, overlay, 0.6, 0)
    
    # Draw thicker boundaries on top for better visibility
    # Ground truth boundary (bright green) - thicker
    gt_boundary = cv2.dilate(ground_truth, kernel, iterations=2) - cv2.erode(ground_truth, kernel, iterations=1)
    boundary_viz[gt_boundary > 0] = [0, 255, 0]
    
    # Prediction boundary (bright yellow) - thicker
    pred_boundary = cv2.dilate(prediction, kernel, iterations=2) - cv2.erode(prediction, kernel, iterations=1)
    boundary_viz[pred_boundary > 0] = [255, 255, 0]
    
    return boundary_viz


def create_qualitative_segmentation_grid(original_image, ground_truth_mask, 
                                         predicted_mask, image_name="Sample"):
    """
    Create comprehensive segmentation visualization grid
    Args:
        original_image: Original image [H, W] or [H, W, 3]
        ground_truth_mask: Ground truth mask [H, W]
        predicted_mask: Predicted mask [H, W]
        image_name: Name/ID of the image
    Returns:
        fig: Matplotlib figure with grid layout
    """
    # Create figure with 1 row, 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Prepare original image
    if len(original_image.shape) == 2:
        img_display = original_image
        cmap = 'gray'
    else:
        img_display = original_image
        cmap = None
    
    # 1. Original Image
    axes[0].imshow(img_display, cmap=cmap)
    axes[0].set_title('Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth Mask
    axes[1].imshow(ground_truth_mask, cmap='gray')
    axes[1].set_title('Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Predicted Mask (Segmentation Mask)
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title('Segmented Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Error Analysis
    error_mask = compute_error_mask(ground_truth_mask, predicted_mask)
    error_viz = visualize_error_analysis(original_image, error_mask)
    axes[3].imshow(error_viz)
    axes[3].set_title('Error Analysis', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # 5. Boundary Details
    boundary_viz = visualize_boundary_details(original_image, ground_truth_mask, predicted_mask)
    axes[4].imshow(boundary_viz)
    axes[4].set_title('Boundary Details', fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    # Add overall title
    fig.suptitle(f'Qualitative Segmentation - {image_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def create_multi_sample_segmentation_grid(samples_list, num_samples=3):
    """
    Create grid showing multiple samples (like in the reference image)
    Args:
        samples_list: List of dictionaries, each containing:
            {
                'original': original image,
                'ground_truth': GT mask,
                'prediction': predicted mask,
                'name': sample name
            }
        num_samples: Number of samples to display (rows)
    Returns:
        fig: Matplotlib figure
    """
    num_samples = min(num_samples, len(samples_list))
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples_list[:num_samples]):
        original = sample['original']
        gt_mask = sample['ground_truth']
        pred_mask = sample['prediction']
        
        # Ensure image is properly formatted
        if len(original.shape) == 2:
            img_display = original
            cmap = 'gray'
        else:
            img_display = original
            cmap = None
        
        # 1. Original Image
        axes[idx, 0].imshow(img_display, cmap=cmap)
        if idx == 0:
            axes[idx, 0].set_title('Image', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # 2. Ground Truth Mask
        axes[idx, 1].imshow(gt_mask, cmap='gray')
        if idx == 0:
            axes[idx, 1].set_title('Mask', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # 3. Predicted Mask
        axes[idx, 2].imshow(pred_mask, cmap='gray')
        if idx == 0:
            axes[idx, 2].set_title('Segmented Mask', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # 4. Error Analysis
        error_mask = compute_error_mask(gt_mask, pred_mask)
        error_viz = visualize_error_analysis(original, error_mask)
        axes[idx, 3].imshow(error_viz)
        if idx == 0:
            axes[idx, 3].set_title('Error Analysis', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
        
        # 5. Boundary Details
        boundary_viz = visualize_boundary_details(original, gt_mask, pred_mask)
        axes[idx, 4].imshow(boundary_viz)
        if idx == 0:
            axes[idx, 4].set_title('Boundary Details', fontsize=12, fontweight='bold')
        axes[idx, 4].axis('off')
    
    plt.tight_layout()
    return fig


def save_segmentation_visualization(samples_list, output_path, num_samples=3, dpi=150):
    """
    Generate and save segmentation visualization
    Args:
        samples_list: List of sample dictionaries
        output_path: Path to save the figure
        num_samples: Number of samples to display
        dpi: DPI for saved figure
    """
    fig = create_multi_sample_segmentation_grid(samples_list, num_samples)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def generate_synthetic_ground_truth(image_shape, num_lesions=1, min_size=20, max_size=60):
    """
    Generate synthetic ground truth for demonstration purposes
    Args:
        image_shape: Tuple of (height, width)
        num_lesions: Number of lesions to generate
        min_size: Minimum lesion size
        max_size: Maximum lesion size
    Returns:
        mask: Binary mask with synthetic lesions
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_lesions):
        # Random lesion position and size
        center_x = np.random.randint(max_size, w - max_size)
        center_y = np.random.randint(max_size, h - max_size)
        radius = np.random.randint(min_size, max_size)
        
        # Draw circular or elliptical lesion
        axes_length = (radius, int(radius * np.random.uniform(0.7, 1.3)))
        angle = np.random.randint(0, 180)
        
        cv2.ellipse(mask, (center_x, center_y), axes_length, angle, 0, 360, 1, -1)
    
    return mask


def simulate_prediction_with_errors(ground_truth, miss_rate=0.2, false_alarm_rate=0.1):
    """
    Simulate prediction with controlled error rates
    Args:
        ground_truth: Ground truth mask
        miss_rate: Probability of missing GT regions (0-1)
        false_alarm_rate: Probability of false alarms (0-1)
    Returns:
        prediction: Simulated prediction mask
    """
    prediction = ground_truth.copy()
    
    # Remove some ground truth regions (false negatives)
    if miss_rate > 0:
        miss_mask = np.random.random(ground_truth.shape) < miss_rate
        prediction[miss_mask & (ground_truth == 1)] = 0
    
    # Add false alarms (false positives)
    if false_alarm_rate > 0:
        fa_mask = np.random.random(ground_truth.shape) < (false_alarm_rate * 0.01)
        prediction[fa_mask & (ground_truth == 0)] = 1
    
    return prediction
