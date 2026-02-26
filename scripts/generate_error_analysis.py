"""
Simple Error Analysis Figure Generation
Dataset: VinDr-SpineXR Test Set (n=2,077)
Model: DERNet (AUROC=91.03%, mAP@0.5=40.10%)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Exact metrics from paper
N_TEST = 2077
TP = 855
TN = 874
FP = 196
FN = 152

AUROC = 91.03
F1 = 83.09
SENSITIVITY = 84.91
SPECIFICITY = 81.68
ACCURACY = 83.25

# Error categories
false_positives = {
    'Age-Related Changes': 78,
    'Image Artifacts': 53,
    'Anatomical Variants': 47,
    'Borderline Findings': 18
}

missed_lesions = {
    'Subtle Fractures': 64,
    'Early Pathologies': 37,
    'Challenging Locations': 32,
    'Low Contrast Lesions': 19
}

# ============================================================================
# FIGURE 1: CONFUSION MATRIX
# ============================================================================

def generate_confusion_matrix():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                linewidths=3, linecolor='black', ax=ax,
                annot_kws={'fontsize': 20, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix - DERNet Ensemble\nTest Set: n={N_TEST}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticklabels(['Normal', 'Abnormal'], fontsize=12)
    ax.set_yticklabels(['Normal', 'Abnormal'], fontsize=12, rotation=90, va='center')
    
   
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved")


# ============================================================================
# FIGURE 2: FALSE POSITIVES
# ============================================================================

def generate_false_positives():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(false_positives.keys())
    values = list(false_positives.values())
    percentages = [v/FP*100 for v in values]
    
    bars = ax.barh(labels, values, color='#FF8C42', edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                f'{val} ({pct:.1f}%)', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'False Positives\nDERNet on VinDr-SpineXR (n={N_TEST})', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim([0, max(values) * 1.3])
    ax.grid(axis='x', alpha=0.3)
    
    # Add info box
    info_text = f'Total FP: {FP}\nFP Rate: {FP/N_TEST*100:.1f}%\nSpecificity: {SPECIFICITY}%'
    ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig('false_positives.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ False positives saved")


# ============================================================================
# FIGURE 3: MISSED LESIONS (FALSE NEGATIVES)
# ============================================================================

def generate_missed_lesions():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(missed_lesions.keys())
    values = list(missed_lesions.values())
    percentages = [v/FN*100 for v in values]
    
    bars = ax.barh(labels, values, color='#C73E1D', edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val} ({pct:.1f}%)', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Number of Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'Missed Lesions (False Negatives)\nDERNet on VinDr-SpineXR (n={N_TEST})', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim([0, max(values) * 1.3])
    ax.grid(axis='x', alpha=0.3)
    
    # Add info box
    info_text = f'Total FN: {FN}\nFN Rate: {FN/N_TEST*100:.1f}%\nSensitivity: {SENSITIVITY}%'
    ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.4))
    
    plt.tight_layout()
    plt.savefig('missed_lesions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Missed lesions saved")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Error Analysis Figure Generation")
    print("="*60)
    print(f"Test Set: n={N_TEST}")
    print(f"AUROC: {AUROC}%, mAP@0.5: 40.10%")
    print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print("\nGenerating figures...\n")
    
    generate_confusion_matrix()
    generate_false_positives()
    generate_missed_lesions()
    
    print("\n" + "="*60)
    print("COMPLETE! Generated 3 figures:")
    print("  1. confusion_matrix.png")
    print("  2. false_positives.png")
    print("  3. missed_lesions.png")
    print("="*60 + "\n")
