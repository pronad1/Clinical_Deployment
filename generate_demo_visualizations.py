"""
Demo Script to Generate Performance Evaluation Visualizations
This script demonstrates how to generate Grad-CAM and Segmentation visualizations standalone
"""

import os
import numpy as np
import cv2
from PIL import Image

# Import our custom modules
from gradcam_generator import save_gradcam_visualization, generate_single_gradcam
from segmentation_visualizer import (
    save_segmentation_visualization,
    generate_synthetic_ground_truth,
    simulate_prediction_with_errors
)


def create_demo_gradcam(test_image_path, models_dict, device='cpu'):
    """
    Create demo Grad-CAM visualization
    
    Args:
        test_image_path: Path to test image
        models_dict: Dictionary of models {name: model}
        device: Device to run on
    """
    print("Generating Grad-CAM visualization...")
    
    import torch
    from torchvision import transforms
    
    # Load and preprocess image
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Prepare for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB
    rgb_image = np.stack([image] * 3, axis=-1)
    pil_image = Image.fromarray(rgb_image)
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Generate Grad-CAM
    output_path = 'demo_gradcam.png'
    save_gradcam_visualization(
        models_dict,
        input_tensor,
        image,
        output_path,
        device=device
    )
    
    print(f"✓ Grad-CAM visualization saved to: {output_path}")
    return output_path


def create_demo_segmentation(test_image_path, num_samples=3):
    """
    Create demo segmentation visualization
    
    Args:
        test_image_path: Path to test image
        num_samples: Number of samples to display
    """
    print("Generating segmentation visualization...")
    
    # Load image
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Generate synthetic data for demonstration
    samples_list = []
    
    for i in range(num_samples):
        # Generate synthetic ground truth
        gt_mask = generate_synthetic_ground_truth(
            image.shape,
            num_lesions=np.random.randint(1, 3),
            min_size=20,
            max_size=50
        )
        
        # Simulate prediction with some errors
        pred_mask = simulate_prediction_with_errors(
            gt_mask,
            miss_rate=0.15,
            false_alarm_rate=0.10
        )
        
        samples_list.append({
            'original': image,
            'ground_truth': gt_mask,
            'prediction': pred_mask,
            'name': f'Sample {i+1}'
        })
    
    # Generate visualization
    output_path = 'demo_segmentation.png'
    save_segmentation_visualization(
        samples_list,
        output_path,
        num_samples=num_samples
    )
    
    print(f"✓ Segmentation visualization saved to: {output_path}")
    return output_path


def main():
    """Main demo function"""
    print("=" * 60)
    print("Performance Evaluation Visualization Demo")
    print("=" * 60)
    
    # Check if test images exist
    test_dir = 'Testing'
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.dcm', '.dicom'))]
    
    if not test_files:
        print("No test files found in Testing directory")
        return
    
    print(f"\nFound {len(test_files)} test files")
    
    # For this demo, we'll show how to use the functions
    # In practice, you'd load your actual models
    
    print("\nTo generate visualizations in your app:")
    print("1. Upload a DICOM file through the web interface")
    print("2. After analysis completes, click 'Generate Grad-CAM Analysis'")
    print("3. Click 'Generate Segmentation Visualization'")
    print("\nThe visualizations will appear below the buttons!")
    
    print("\n" + "=" * 60)
    print("To run the app:")
    print("  python app.py")
    print("Then navigate to: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
