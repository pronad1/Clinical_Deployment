"""
Generate preview visualization for qualitative segmentation
"""
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import pydicom
matplotlib.use('Agg')

# Load test image
dcm = pydicom.dcmread('Testing/normal.dicom')
original_image = dcm.pixel_array.astype(np.uint8)

# Normalize if needed
if original_image.max() > 255:
    original_image = ((original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255).astype(np.uint8)

h, w = original_image.shape

# Create dummy ground truth and prediction masks
ground_truth = np.zeros((h, w), dtype=np.uint8)
prediction = np.zeros((h, w), dtype=np.uint8)

# Draw GT box
gt_x1, gt_y1 = int(w * 0.3), int(h * 0.3)
gt_x2, gt_y2 = int(w * 0.7), int(h * 0.7)
ground_truth[gt_y1:gt_y2, gt_x1:gt_x2] = 1

# Draw Prediction box (slightly offset to create errors)
pred_x1, pred_y1 = gt_x1 + 15, gt_y1 + 15
pred_x2, pred_y2 = gt_x2 + 15, gt_y2 + 15
prediction[pred_y1:pred_y2, pred_x1:pred_x2] = 1

# Create 5-column visualization
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.patch.set_facecolor('white')

# Column 1: Original Image
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Column 2: Ground Truth
axes[1].imshow(ground_truth, cmap='gray')
axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Column 3: Prediction
axes[2].imshow(prediction, cmap='gray')
axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
axes[2].axis('off')

# Column 4: Error Analysis (NEW IMPROVED VERSION - BLACK BACKGROUND)
error_mask = np.zeros_like(ground_truth, dtype=np.uint8)
error_mask[(ground_truth == 1) & (prediction == 0)] = 1  # False Negatives
error_mask[(ground_truth == 0) & (prediction == 1)] = 2  # False Positives

# Create pure black background
error_vis = np.zeros((h, w, 3), dtype=np.uint8)

# Add errors with BRIGHT PURE colors
error_vis[error_mask == 1] = [0, 255, 0]  # Pure GREEN for False Negatives
error_vis[error_mask == 2] = [255, 0, 0]  # Pure RED for False Positives

# If no errors, show darkened background
if not np.any(error_mask > 0):
    error_vis = (original_image[:, :, np.newaxis] * 0.8).astype(np.uint8)
    error_vis = np.repeat(error_vis, 3, axis=2)

axes[3].imshow(error_vis)
axes[3].set_title('Error Analysis\n(Green=FN, Red=FP)', fontsize=14, fontweight='bold')
axes[3].axis('off')

# Column 5: Boundary Details (NEW IMPROVED VERSION - DARK BACKGROUND)
# Create 40% darkened background
boundary_vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
boundary_vis = (boundary_vis * 0.6).astype(np.uint8)

# Create thicker overlays
gt_mask_thick = cv2.dilate(ground_truth, np.ones((3,3), np.uint8), iterations=2)
pred_mask_thick = cv2.dilate(prediction, np.ones((3,3), np.uint8), iterations=2)

# Blend overlays with 60% opacity
overlay = np.zeros_like(boundary_vis)
overlay[ground_truth == 1] = [0, 200, 0]  # Green regions for GT
overlay[prediction == 1] = [180, 255, 80]  # Yellow-green for predictions

# Blend with background (60% overlay, 40% background)
boundary_vis = cv2.addWeighted(boundary_vis, 0.4, overlay, 0.6, 0)

# Draw THICK BRIGHT boundaries on top
gt_boundary = cv2.dilate(ground_truth, np.ones((3,3), np.uint8)) - ground_truth
pred_boundary = cv2.dilate(prediction, np.ones((3,3), np.uint8)) - prediction

boundary_vis[gt_boundary > 0] = [0, 255, 0] # BRIGHT GREEN for GT boundary
boundary_vis[pred_boundary > 0] = [255, 255, 0]  # BRIGHT YELLOW for Pred boundary

axes[4].imshow(boundary_vis)
axes[4].set_title('Boundary Details\n(Green=GT, Yellow=Pred)', fontsize=14, fontweight='bold')
axes[4].axis('off')

plt.tight_layout()

# Save as image
output_path = 'qualitative_preview.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✓ Saved preview to: {output_path}')

# Convert to base64 for drawio
import io
import base64

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

# Save base64
with open('qualitative_preview_base64.txt', 'w') as f:
    f.write(img_base64)

print(f'✓ Generated base64 encoding ({len(img_base64)} chars)')
print('\nPreview shows:')
print('- Column 4 (Error Analysis): BLACK background with BRIGHT green/red errors')
print('- Column 5 (Boundary Details): DARKER background with THICKER, BRIGHTER boundaries')

plt.close()
