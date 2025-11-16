"""
NutriPredict - Sistem Prediksi Stunting dengan ANN/MLP
Training Script dengan Evaluasi Lengkap (Confusion Matrix, Handling Imbalance)
Dataset: Kategori stunting bayi (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. LOAD & EXPLORASI DATA ====================
print("=" * 60)
print("STEP 1: LOAD & EKSPLORASI DATA")
print("=" * 60)

# Load dataset (sesuaikan path)
df = pd.read_csv('child_data_rev.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 Columns: {df.columns.tolist()}")
print(f"\n🔍 First 5 rows:\n{df.head()}")
print(f"\n📈 Info Dataset:")
print(df.info())
print(f"\n📊 Statistical Summary:")
print(df.describe())
print(f"\n❓ Missing Values:")
print(df.isnull().sum())

# ==================== 2. DATA PREPROCESSING ====================
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# Handle missing values
df = df.dropna()

# Identifikasi kolom kategorikal dan numerikal
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Hapus target dari numerical_cols jika ada
target_col = 'Status'  # Sesuaikan dengan nama kolom target
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"\n📝 Categorical columns: {categorical_cols}")
print(f"\n🔢 Numerical columns: {numerical_cols}")

# Label Encoding untuk categorical features
label_encoders = {}
for col in categorical_cols:
    if col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Encode target variable
if target_col in df.columns:
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    print(f"\n🎯 Target Classes: {le_target.classes_}")
    print(f"   Encoded as: {list(range(len(le_target.classes_)))}")

# ==================== 3. CEK DATA IMBALANCE ====================
print("\n" + "=" * 60)
print("STEP 3: CEK DATA IMBALANCE")
print("=" * 60)

class_distribution = df[target_col].value_counts().sort_index()
print(f"\n📊 Class Distribution (BEFORE Balancing):")
for idx, count in class_distribution.items():
    class_name = le_target.inverse_transform([idx])[0]
    percentage = (count / len(df)) * 100
    print(f"   Class {idx} ({class_name}): {count} samples ({percentage:.2f}%)")

# Visualisasi distribusi kelas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
class_distribution.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Class Distribution (Before Balancing)')
plt.xlabel('Stunting Status')
plt.ylabel('Count')
plt.xticks(rotation=0)

# ==================== 4. SPLIT DATA & HANDLE IMBALANCE ====================
print("\n" + "=" * 60)
print("STEP 4: SPLIT DATA & HANDLE IMBALANCE")
print("=" * 60)

# Pisahkan features dan target
X = df.drop(columns=[target_col])
y = df[target_col]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️ Train set: {X_train.shape[0]} samples")
print(f"✂️ Test set: {X_test.shape[0]} samples")

# Handle imbalance dengan SMOTE
print("\n🔄 Applying SMOTE for balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

balanced_distribution = pd.Series(y_train_balanced).value_counts().sort_index()
print(f"\n📊 Class Distribution (AFTER Balancing):")
for idx, count in balanced_distribution.items():
    class_name = le_target.inverse_transform([idx])[0]
    percentage = (count / len(y_train_balanced)) * 100
    print(f"   Class {idx} ({class_name}): {count} samples ({percentage:.2f}%)")

# Plot distribusi setelah balancing
plt.subplot(1, 2, 2)
balanced_distribution.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title('Class Distribution (After SMOTE)')
plt.xlabel('Stunting Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: class_distribution.png")

# ==================== 5. FEATURE SCALING ====================
print("\n" + "=" * 60)
print("STEP 5: FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled using StandardScaler")
print(f"   Train shape: {X_train_scaled.shape}")
print(f"   Test shape: {X_test_scaled.shape}")

# Convert target ke categorical untuk keras
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train_balanced, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ==================== 6. BUILD MLP MODEL (ANN MURNI) ====================
print("\n" + "=" * 60)
print("STEP 6: BUILD MLP MODEL (ANN MURNI)")
print("=" * 60)

# Arsitektur MLP
input_dim = X_train_scaled.shape[1]

model = keras.Sequential([
    # Input Layer
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    
    # Hidden Layer 1
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # Hidden Layer 2
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    # Output Layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Model Architecture:")
model.summary()

# ==================== 7. TRAIN MODEL ====================
print("\n" + "=" * 60)
print("STEP 7: TRAIN MODEL")
print("=" * 60)

# Class weights untuk handling imbalance tambahan
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_balanced),
    y=y_train_balanced
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n⚖️ Class Weights: {class_weight_dict}")

# Training
history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# ==================== 8. EVALUASI MODEL ====================
print("\n" + "=" * 60)
print("STEP 8: EVALUASI MODEL")
print("=" * 60)

# Prediksi
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n📊 CLASSIFICATION REPORT:")
print("=" * 60)
target_names = [le_target.inverse_transform([i])[0] for i in range(num_classes)]
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# Precision, Recall, F1-Score per class
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None
)

print("\n📈 DETAILED METRICS PER CLASS:")
print("=" * 60)
for i in range(num_classes):
    class_name = le_target.inverse_transform([i])[0]
    print(f"\n{class_name.upper()}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")

# ==================== 9. CONFUSION MATRIX ====================
print("\n" + "=" * 60)
print("STEP 9: CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print("\n🎯 Confusion Matrix:")
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - NutriPredict ANN Model', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: confusion_matrix.png")

# Analisis False Positives & False Negatives
print("\n⚠️ ANALISIS FALSE POSITIVES & FALSE NEGATIVES:")
print("=" * 60)
for i in range(num_classes):
    class_name = target_names[i]
    true_positives = cm[i, i]
    false_positives = cm[:, i].sum() - true_positives
    false_negatives = cm[i, :].sum() - true_positives
    
    print(f"\n{class_name.upper()}:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    
    if i == 2:  # High risk class (paling penting)
        print(f"\n  ⚠️ CRITICAL: {false_negatives} anak berisiko TINGGI diprediksi lebih rendah!")
        print(f"  (Ini berbahaya karena mereka butuh intervensi segera)")

# ==================== 10. TRAINING HISTORY ====================
print("\n" + "=" * 60)
print("STEP 10: TRAINING HISTORY")
print("=" * 60)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: training_history.png")

# ==================== 11. SAVE MODEL & ARTIFACTS ====================
print("\n" + "=" * 60)
print("STEP 11: SAVE MODEL & ARTIFACTS")
print("=" * 60)

# Save model
model.save('nutripredict_model.h5')
print("\n💾 Saved: nutripredict_model.h5")

# Save scaler & encoders
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')
print("💾 Saved: scaler.pkl, label_encoders.pkl, target_encoder.pkl")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("💾 Saved: feature_names.pkl")

# Save evaluation metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision.tolist(),
    'recall': recall.tolist(),
    'f1_score': f1.tolist(),
    'confusion_matrix': cm.tolist(),
    'target_names': target_names
}
joblib.dump(metrics, 'metrics.pkl')
print("💾 Saved: metrics.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Final Results:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Avg Precision: {precision.mean():.4f}")
print(f"   Avg Recall: {recall.mean():.4f}")
print(f"   Avg F1-Score: {f1.mean():.4f}")
print("\n🎯 Next Step: Run streamlit_app.py untuk demo prediksi!")

