"""
Test LIME Explainability Implementation
Quick test to verify LIME works with spine images
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

# Import LIME explainer
from lime_explainer import generate_lime_grid, generate_lime_comparison, FastLIMEExplainer


def test_lime_implementation():
    """Test LIME implementation with spine image"""
    
    print("="*70)
    print("LIME Explainability Test")
    print("="*70)
    
    # Set device
    device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    # Load models
    print("\n📦 Loading models...")
    models = {}
    
    try:
        # Load DenseNet121
        densenet = timm.create_model('densenet121', pretrained=False, num_classes=1)
        densenet_path = os.path.join('ensemble output', 'densenet121_balanced', 'model_best.pth')
        if os.path.exists(densenet_path):
            checkpoint = torch.load(densenet_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            densenet.load_state_dict(state_dict)
            densenet.eval()
            models['densenet121'] = densenet
            print("  ✓ DenseNet121 loaded")
        
        # Load ResNet50
        resnet = timm.create_model('resnet50', pretrained=False, num_classes=1)
        resnet_path = os.path.join('ensemble output', 'resnet50_optimized', 'model_best.pth')
        if os.path.exists(resnet_path):
            checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            resnet.load_state_dict(state_dict)
            resnet.eval()
            models['resnet50'] = resnet
            print("  ✓ ResNet50 loaded")
        
        # Load EfficientNetV2
        efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=1)
        efficientnet_path = os.path.join('ensemble output', 'tf_efficientnetv2_s_optimized', 'model_best.pth')
        if os.path.exists(efficientnet_path):
            checkpoint = torch.load(efficientnet_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            efficientnet.load_state_dict(state_dict)
            efficientnet.eval()
            models['efficientnet'] = efficientnet
            print("  ✓ EfficientNetV2 loaded")
            
    except Exception as e:
        print(f"  ✗ Error loading models: {e}")
        return False
    
    if len(models) == 0:
        print("\n❌ No models loaded. Exiting.")
        return False
    
    print(f"\n✓ {len(models)} models loaded successfully")
    
    # Load test image
    print("\n📸 Loading test image...")
    test_image_path = os.path.join('Testing', 'ab.dicom')
    
    if not os.path.exists(test_image_path):
        test_dir = 'Testing'
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith(('.dicom', '.dcm'))]
            if test_files:
                test_image_path = os.path.join(test_dir, test_files[0])
                print(f"  Using: {test_image_path}")
            else:
                print("  ✗ No test images found")
                return False
        else:
            print("  ✗ Testing directory not found")
            return False
    
    # Read DICOM
    try:
        import pydicom
        ds = pydicom.dcmread(test_image_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Normalize
        pixel_array = ((pixel_array - pixel_array.min()) / 
                      (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        print(f"  ✓ Image loaded: {pixel_array.shape}")
        
    except Exception as e:
        print(f"  ✗ Error loading image: {e}")
        return False
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB
    if len(pixel_array.shape) == 2:
        rgb_image = np.stack([pixel_array] * 3, axis=-1)
    else:
        rgb_image = pixel_array
    
    pil_image = Image.fromarray(rgb_image)
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Ensemble weights
    ensemble_weights = {
        'densenet121': 0.42,
        'efficientnet': 0.32,
        'resnet50': 0.26
    }
    
    # Test Grid visualization
    print("\n" + "="*70)
    print("Generating LIME Grid Visualization")
    print("="*70)
    
    try:
        fig = generate_lime_grid(
            models,
            input_tensor,
            pixel_array,
            device,
            ensemble_weights=ensemble_weights,
            n_samples=50,
            n_segments=10
        )
        
        output_path = 'test_lime_grid.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\n✅ Grid visualization saved: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating grid: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Comparison visualization
    print("\n" + "="*70)
    print("Generating LIME Comparison Visualization")
    print("="*70)
    
    try:
        fig = generate_lime_comparison(
            models,
            input_tensor,
            pixel_array,
            device,
            ensemble_weights=ensemble_weights,
            n_samples=50,
            n_segments=10
        )
        
        output_path = 'test_lime_comparison.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"\n✅ Comparison visualization saved: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating comparison: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nGenerated files:")
    print("  • test_lime_grid.png")
    print("  • test_lime_comparison.png")
    print("\nReady to use in web application!")
    
    return True


if __name__ == '__main__':
    success = test_lime_implementation()
    sys.exit(0 if success else 1)
