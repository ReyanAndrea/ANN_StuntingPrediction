"""
NutriPredict - Detailed Model Evaluation
Script untuk evaluasi mendalam model yang sudah di-train
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from tensorflow import keras
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("NUTRIPREDICT - DETAILED MODEL EVALUATION")
print("=" * 70)

# ==================== 1. LOAD MODEL & TEST DATA ====================
print("\n📂 STEP 1: LOAD MODEL & ARTIFACTS")
print("-" * 70)

try:
    model = keras.models.load_model('nutripredict_model.h5')
    scaler = joblib.load('scaler.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    metrics = joblib.load('metrics.pkl')
    print("✅ Model and artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Please run train_model.py first!")
    exit()

# Load test data (you need to save this from training script)
# For demo, we'll use the metrics that were saved
cm = np.array(metrics['confusion_matrix'])
target_names = metrics['target_names']
accuracy = metrics['accuracy']
precision = np.array(metrics['precision'])
recall = np.array(metrics['recall'])
f1_score = np.array(metrics['f1_score'])

num_classes = len(target_names)

print(f"\n📊 Model Info:")
print(f"   Classes: {target_names}")
print(f"   Number of classes: {num_classes}")
print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ==================== 2. CONFUSION MATRIX DETAILED ANALYSIS ====================
print("\n\n🎯 STEP 2: CONFUSION MATRIX DETAILED ANALYSIS")
print("-" * 70)

print("\n📊 Confusion Matrix:")
print(cm)

# Calculate metrics from confusion matrix
print("\n📈 Per-Class Metrics from Confusion Matrix:")
print("=" * 70)

for i, class_name in enumerate(target_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n{class_name.upper()}:")
    print(f"  True Positives (TP):  {tp:>4}")
    print(f"  False Positives (FP): {fp:>4}")
    print(f"  False Negatives (FN): {fn:>4}")
    print(f"  True Negatives (TN):  {tn:>4}")
    print(f"  ---")
    print(f"  Sensitivity (Recall):  {sensitivity:.4f}")
    print(f"  Specificity:           {specificity:.4f}")
    print(f"  Precision (PPV):       {ppv:.4f}")
    print(f"  Negative Pred Value:   {npv:.4f}")
    
    # Warning for high-risk class
    if class_name.lower() in ['tinggi', 'high', 'severe']:
        if fn > 0:
            print(f"\n  ⚠️ CRITICAL WARNING:")
            print(f"  {fn} children at HIGH RISK were predicted as lower risk!")
            print(f"  This means they might not receive urgent intervention.")
            print(f"  Sensitivity/Recall: {sensitivity:.4f} ({sensitivity*100:.1f}%)")

# ==================== 3. CLASSIFICATION REPORT ====================
print("\n\n📋 STEP 3: DETAILED CLASSIFICATION REPORT")
print("-" * 70)

# Create a formatted classification report
print("\n" + "=" * 70)
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
print("=" * 70)

for i, class_name in enumerate(target_names):
    support = cm[i, :].sum()
    print(f"{class_name:<15} {precision[i]:>8.4f}    {recall[i]:>8.4f}    {f1_score[i]:>8.4f}    {support:>8}")

print("=" * 70)
print(f"{'Accuracy':<15} {'':<12} {'':<12} {accuracy:>8.4f}    {cm.sum():>8}")
print(f"{'Macro Avg':<15} {precision.mean():>8.4f}    {recall.mean():>8.4f}    {f1_score.mean():>8.4f}    {cm.sum():>8}")

# Weighted average
total_support = cm.sum()
weighted_precision = np.sum(precision * cm.sum(axis=1)) / total_support
weighted_recall = np.sum(recall * cm.sum(axis=1)) / total_support
weighted_f1 = np.sum(f1_score * cm.sum(axis=1)) / total_support

print(f"{'Weighted Avg':<15} {weighted_precision:>8.4f}    {weighted_recall:>8.4f}    {weighted_f1:>8.4f}    {cm.sum():>8}")
print("=" * 70)

# ==================== 4. CONFUSION MATRIX HEATMAP WITH PERCENTAGES ====================
print("\n\n📊 STEP 4: CONFUSION MATRIX VISUALIZATION")
print("-" * 70)

# Normalize confusion matrix for percentages
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Counts
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={'label': 'Count'},
    ax=axes[0],
    square=True
)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

# Plot 2: Percentages
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Greens',
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={'label': 'Percentage'},
    ax=axes[1],
    square=True
)
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: confusion_matrix_detailed.png")

# ==================== 5. ERROR ANALYSIS ====================
print("\n\n❌ STEP 5: ERROR ANALYSIS")
print("-" * 70)

# Misclassification rates
print("\n📉 Misclassification Matrix:")
print("(Shows where predictions went wrong)")
print("\nTrue → Predicted:")

misclassification = pd.DataFrame(cm, 
                                 index=[f"True {name}" for name in target_names],
                                 columns=[f"Pred {name}" for name in target_names])
print(misclassification)

# Most common errors
print("\n⚠️ Most Common Misclassifications:")
errors = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            errors.append({
                'True Class': target_names[i],
                'Predicted As': target_names[j],
                'Count': cm[i, j],
                'Percentage': f"{(cm[i, j] / cm[i, :].sum() * 100):.2f}%"
            })

errors_df = pd.DataFrame(errors).sort_values('Count', ascending=False)
if len(errors_df) > 0:
    print(errors_df.head(10).to_string(index=False))
else:
    print("No misclassifications! (Perfect model)")

# ==================== 6. METRICS COMPARISON ====================
print("\n\n📊 STEP 6: METRICS COMPARISON")
print("-" * 70)

# Create comparison plot
metrics_data = pd.DataFrame({
    'Class': target_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score
})

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(target_names))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics by Class', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(target_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: metrics_comparison.png")

# ==================== 7. SUMMARY & RECOMMENDATIONS ====================
print("\n\n" + "=" * 70)
print("📋 EVALUATION SUMMARY")
print("=" * 70)

print(f"\n✅ Overall Performance:")
print(f"   Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Avg Precision:  {precision.mean():.4f}")
print(f"   Avg Recall:     {recall.mean():.4f}")
print(f"   Avg F1-Score:   {f1_score.mean():.4f}")

print(f"\n🎯 Best Performing Class:")
best_class = target_names[f1_score.argmax()]
print(f"   {best_class}: F1-Score = {f1_score.max():.4f}")

print(f"\n⚠️ Needs Improvement:")
worst_class = target_names[f1_score.argmin()]
print(f"   {worst_class}: F1-Score = {f1_score.min():.4f}")

# Check for critical issues
critical_issues = []

for i, class_name in enumerate(target_names):
    if recall[i] < 0.7:
        critical_issues.append(f"Low recall for {class_name} ({recall[i]:.4f})")
    if precision[i] < 0.7:
        critical_issues.append(f"Low precision for {class_name} ({precision[i]:.4f})")

if critical_issues:
    print(f"\n🚨 Critical Issues Found:")
    for issue in critical_issues:
        print(f"   - {issue}")
else:
    print(f"\n✅ No critical issues detected!")

print(f"\n💡 Recommendations:")
if accuracy > 0.85:
    print(f"   ✅ Model performance is good!")
else:
    print(f"   ⚠️ Consider improving model (try different architecture, more data)")

if any(recall < 0.75):
    print(f"   ⚠️ Some classes have low recall - may need more training data or rebalancing")

if any(precision < 0.75):
    print(f"   ⚠️ Some classes have low precision - model may be too aggressive in predictions")

# Check if high-risk class is well detected
high_risk_idx = -1
for i, name in enumerate(target_names):
    if name.lower() in ['tinggi', 'high', 'severe']:
        high_risk_idx = i
        break

if high_risk_idx >= 0:
    hr_recall = recall[high_risk_idx]
    print(f"\n🎯 High-Risk Class Detection:")
    if hr_recall >= 0.85:
        print(f"   ✅ Excellent! Recall = {hr_recall:.4f}")
    elif hr_recall >= 0.75:
        print(f"   ⚠️ Good, but can be improved. Recall = {hr_recall:.4f}")
    else:
        print(f"   🚨 CRITICAL! Low recall = {hr_recall:.4f}")
        print(f"   Many high-risk children might be missed!")

print("\n" + "=" * 70)
print("DETAILED EVALUATION COMPLETED! 🎉")
print("=" * 70)
print("\n📊 Generated files:")
print("   - confusion_matrix_detailed.png")
print("   - metrics_comparison.png")