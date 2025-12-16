# 🍼 NutriPredict - Sistem Prediksi Stunting Berbasis ANN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Sistem Prediksi Risiko Stunting untuk Intervensi Dini**

[Demo](#-demo) • [Instalasi](#-instalasi) • [Dokumentasi](#-dokumentasi) • [Tim](#-tim-pengembang)

</div>

---

## 📋 Deskripsi Proyek

**NutriPredict** adalah sistem prediksi stunting berbasis **Artificial Neural Network (ANN)** yang dikembangkan untuk membantu tenaga kesehatan dalam melakukan deteksi dini risiko stunting pada anak balita. Sistem ini menggunakan Multi-Layer Perceptron (MLP) untuk mengklasifikasi status gizi anak ke dalam 4 kategori: Normal, Overweight, Underweight, dan Stunting.

### 🎯 Tujuan Proyek

1. Membangun sistem prediksi stunting dengan akurasi tinggi menggunakan ANN
2. Memberikan rekomendasi intervensi gizi yang personal berdasarkan hasil prediksi
3. Membantu tenaga kesehatan melakukan deteksi dini secara lebih efektif
4. Mendukung program nasional penurunan stunting hingga target 14%

### 📚 Latar Belakang

Menurut data Kementerian Kesehatan RI (2023), prevalensi stunting di Indonesia masih sebesar **21,5%**. Masalah utama adalah intervensi yang sering terlambat karena:

- Pencegahan masih dilakukan secara manual tanpa analisis prediktif
- Data gizi belum dimanfaatkan optimal untuk deteksi risiko lebih awal
- Kurangnya sistem terintegrasi untuk screening massal

**NutriPredict** hadir sebagai solusi dengan memanfaatkan teknologi AI untuk prediksi dini dan rekomendasi intervensi personal.

---

## ✨ Fitur Utama

### 🧠 Model Prediksi
- **Arsitektur**: Multi-Layer Perceptron (MLP) 3 hidden layers
- **Akurasi**: 80% pada test set
- **Precision Stunting**: 100% (tidak ada false alarm pada kasus kritis)
- **Recall Overweight & Underweight**: 100% (semua kasus terdeteksi)

### 📊 Dashboard Interaktif
- Input data anak secara real-time
- Prediksi dengan confidence score
- Visualisasi probabilitas per kategori
- Rekomendasi intervensi personal
- Evaluasi performa model lengkap

### ⚖️ Data Processing
- Handling missing values
- SMOTE untuk balancing imbalanced data
- Feature scaling dengan StandardScaler
- Label encoding untuk kategorikal features
- Stratified train-test split

### 📈 Model Evaluation
- Confusion Matrix dengan analisis detail
- Classification Report (Precision, Recall, F1-Score)
- Training & Validation curves
- Per-class performance analysis

---

## 🛠️ Tech Stack

| Kategori | Teknologi | Versi |
|----------|-----------|-------|
| **Language** | Python | 3.10+ |
| **Deep Learning** | TensorFlow/Keras | 2.13.0 |
| **Web Framework** | Streamlit | 1.28.0 |
| **Data Processing** | Pandas | 2.0.3 |
| **Numerical** | NumPy | 1.24.3 |
| **ML Utilities** | Scikit-learn | 1.3.0 |
| **Imbalance Handling** | Imbalanced-learn | 0.10.1 |
| **Visualization** | Matplotlib | 3.7.2 |
| **Statistical Plots** | Seaborn | 0.12.2 |
| **Serialization** | Joblib | 1.3.1 |

---

## 📊 Dataset

### Sumber Data
**Indonesian Children Medical & Food Nutrition** - Data dari berbagai Pusat Kesehatan Masyarakat (Puskesmas)

### Karakteristik
- **Total Sampel**: 100 anak balita
- **Usia**: 1-5 tahun
- **Split**: 80% Training (80 samples) / 20% Testing (20 samples)
- **Split Method**: Stratified (mempertahankan proporsi kelas)

### Fitur Input (4 Features)

| Fitur | Tipe | Deskripsi | Range/Value |
|-------|------|-----------|-------------|
| **Sex** | Categorical | Jenis kelamin | Male / Female |
| **Age** | Numerical | Usia anak | 1-5 tahun |
| **Height** | Numerical | Tinggi badan | cm |
| **Weight** | Numerical | Berat badan | kg |

### Target Output (1 Feature)

| Kategori | Definisi |
|----------|----------|
| **Normal** | Tinggi dan berat sesuai standar WHO |
| **Overweight** | Berat berlebih dibanding standar |
| **Underweight** | Berat kurang dari standar |
| **Stunting** | Tinggi kurang dari standar WHO untuk usia |

### Distribusi Data

**Original Dataset (Balanced):**
- Normal: 25 samples (25%)
- Overweight: 25 samples (25%)
- Stunting: 25 samples (25%)
- Underweight: 25 samples (25%)

**After SMOTE (Training Set):**
Dataset training menggunakan SMOTE untuk memastikan model tidak bias terhadap kelas tertentu.

---

## 🏗️ Arsitektur Model

### Model Architecture

```python
Model: Multi-Layer Perceptron (MLP)
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
input (InputLayer)          (None, 4)                 0         
_________________________________________________________________
dense_1 (Dense)             (None, 128)               640       
dropout_1 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)             (None, 64)                8256      
dropout_2 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)             (None, 32)                2080      
dropout_3 (Dropout)         (None, 32)                0         
_________________________________________________________________
output (Dense)              (None, 4)                 132       
=================================================================
Total params: 11,108 (43.39 KB)
Trainable params: 11,108 (43.39 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

### Layer Configuration

| Layer | Neurons | Activation | Dropout | Purpose |
|-------|---------|------------|---------|---------|
| Input | 4 | - | - | Input features |
| Hidden 1 | 128 | ReLU | 0.3 | Feature extraction |
| Hidden 2 | 64 | ReLU | 0.3 | Pattern recognition |
| Hidden 3 | 32 | ReLU | 0.2 | Feature refinement |
| Output | 4 | Softmax | - | Classification |

### Hyperparameters

| Parameter         | Nilai                     |
|-------------------|---------------------------|
| Optimizer         | Adam (lr = 0.001)         |
| Loss              | Categorical Crossentropy  |
| Epochs            | 150                       |
| Batch Size        | 32                        |
| Validation Split  | 0.2                       |


### Training Configuration

- **Optimizer**: Adam (Adaptive Learning Rate)
- **Loss Function**: Categorical Crossentropy
- **Regularization**: Dropout (0.3, 0.3, 0.2)
- **Class Balancing**: SMOTE + Class Weights
- **Early Stopping**: Monitoring validation loss

---

## 💻 Instalasi

### Prerequisites

- Python 3.10 atau lebih tinggi
- pip (Python package manager)
- Git

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/ReyanAndrea/ANN_StuntingPrediction.git
cd ANN_StuntingPrediction

# 2. Install dependencies
pip install -r requirements.txt

# Atau install manual
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow streamlit joblib

# 3. Verifikasi instalasi
python --version
python -c "import tensorflow; print(tensorflow.__version__)"
streamlit --version
```

### Dependencies

Buat file `requirements.txt`:

```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.10.1
tensorflow==2.13.0
streamlit==1.28.0
joblib==1.3.1
```

---

## 🚀 Cara Penggunaan

### 1️⃣ Training Model

```bash
python train_model.py
```

**Output yang dihasilkan:**
```
├── nutripredict_model.h5      # Trained model weights
├── scaler.pkl                  # StandardScaler untuk normalisasi
├── label_encoders.pkl          # Encoder untuk Sex
├── target_encoder.pkl          # Encoder untuk Status
├── feature_names.pkl           # Nama fitur
├── metrics.pkl                 # Performance metrics
├── confusion_matrix.png        # Confusion matrix
├── training_history.png        # Training curves
└── class_distribution.png      # Data distribution
```

### 2️⃣ Menjalankan Web App

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

### 3️⃣ Melakukan Prediksi

**Via Web Interface:**

1. **Input Data Anak**
   - Jenis kelamin: Male/Female
   - Usia: 1-5 tahun
   - Tinggi badan (cm)
   - Berat badan (kg)

2. **Klik "🔮 Prediksi Status Gizi"**

3. **Hasil Prediksi**
   - Status gizi: Normal/Overweight/Underweight/Stunting
   - Confidence score (%)
   - Visualisasi probabilitas
   - Rekomendasi intervensi

**Via Python Script:**

```python
import joblib
import numpy as np

# Load model dan preprocessing tools
model = joblib.load('nutripredict_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoders.pkl')

# Input data
sex = 'Male'  # Male/Female
age = 3  # tahun
height = 85  # cm
weight = 12  # kg

# Preprocess
sex_encoded = label_encoder.transform([sex])[0]
features = np.array([[sex_encoded, age, height, weight]])
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
class_names = ['Normal', 'Overweight', 'Stunting', 'Underweight']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Status: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 📊 Performa Model

### Overall Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 80.00% |
| **Weighted Precision** | 81.71% |
| **Weighted Recall** | 80.00% |
| **Weighted F1-Score** | 79.99% |
| **Macro Precision** | 81.43% |
| **Macro Recall** | 79.17% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 80.00% | 66.67% | 72.73% | 6 |
| Overweight | 85.71% | 100.00% | 92.31% | 6 |
| Stunting | 100.00% | 75.00% | 85.71% | 4 |
| Underweight | 60.00% | 100.00% | 75.00% | 4 |

### Confusion Matrix

```
Predicted →       Normal  Overweight  Stunting  Underweight
Actual ↓
Normal               5         1          0          0
Overweight           2         4          0          0
Stunting             0         0          3          1
Underweight          0         0          0          4
```

### Key Insights

✅ **Strengths:**
- **80% overall accuracy** - Sangat baik untuk medical screening
- **100% precision stunting** - Tidak ada false alarm pada kasus kritis
- **100% recall overweight & underweight** - Semua kasus terdeteksi
- **Strong diagonal** - 16/20 prediksi benar

⚠️ **Areas for Improvement:**
- **Stunting recall 75%** - 1 dari 4 kasus terlewat (diprediksi underweight)
- **Underweight precision 60%** - Ada overlap dengan kelas normal
- **Dataset size** - Perlu lebih banyak sampel untuk generalisasi

### Training Performance

| Metric | Epoch 1 | Epoch 100 | Final |
|--------|---------|-----------|-------|
| Training Accuracy | ~25% | ~95% | 95% |
| Validation Accuracy | ~37% | ~82% | 82% |
| Training Loss | ~1.38 | ~0.12 | 0.12 |
| Validation Loss | ~1.36 | ~0.45 | 0.45 |

**Interpretasi:**
- Model belajar dengan baik dan konsisten
- Slight overfitting (gap 13% antara train-val accuracy)
- Dropout dan class weights membantu regularization
- Validation accuracy 82% cukup baik untuk screening

---

## 📁 Struktur Proyek

```
ANN_StuntingPrediction/
├── child_data_rev.csv              # Dataset anak
├── nutripredict_model.h5           # Trained model
├── scaler.pkl                      # Feature scaler
├── label_encoders.pkl              # Categorical encoders
├── target_encoder.pkl              # Target encoder
├── feature_names.pkl               # Feature metadata
├── metrics.pkl                     # Performance metrics
├── confusion_matrix.png            # Confusion matrix plot
├── training_history.png            # Training curves
├── class_distribution.png          # Data distribution
├── train_model.py                  # Model training script
├── preprocessing.py                # Data preprocessing
├── model.py                        # Model architecture
├── evaluation.py                   # Model evaluation
├── streamlit_app.py                # Web application
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore file
```

---

## 🔬 Metodologi

### 1. Data Preprocessing Pipeline

```python
# Step 1: Data Cleaning
df = pd.read_csv('child_data_rev.csv')
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates

# Step 2: Label Encoding
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

le_status = LabelEncoder()
y = le_status.fit_transform(df['Status'])

# Step 3: Train-Test Split (Stratified)
X = df[['Sex', 'Age', 'Height', 'Weight']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: SMOTE (Handling Imbalance)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 6: One-Hot Encoding (Target)
y_train_cat = to_categorical(y_train_balanced)
y_test_cat = to_categorical(y_test)
```

### 2. Model Building

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(4,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Model Training

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)
```

### 4. Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Metrics
print(classification_report(y_test, y_pred_classes))
print(confusion_matrix(y_test, y_pred_classes))
```

---

## 🏥 Implementasi Klinis

### Protokol Penggunaan Sistem

#### 1. Prediksi Stunting (Precision 100%, Recall 75%)
- ✅ **Action**: Setiap prediksi stunting → rujukan DARURAT
- ✅ **Confidence**: Sangat tinggi (no false alarm)
- ⚠️ **Warning**: 25% kasus bisa missed (diprediksi underweight)
- 📋 **Protocol**: Intervensi emergency, rujuk spesialis anak + ahli gizi

#### 2. Prediksi Overweight (Precision 85.71%, Recall 100%)
- ✅ **Action**: Konsultasi gizi, monitor pola makan
- ✅ **Confidence**: Tinggi, semua kasus terdeteksi
- 📋 **Protocol**: Edukasi orang tua, kurangi gula/lemak, aktivitas fisik

#### 3. Prediksi Underweight (Precision 60%, Recall 100%)
- ⚠️ **Action**: WAJIB screening stunting lanjutan
- ⚠️ **Confidence**: Sedang (40% false positive dari normal)
- ⚠️ **Critical**: Mungkin ada stunting yang masuk kategori ini
- 📋 **Protocol**: Evaluasi tinggi badan, cek standar WHO, rujuk medis

#### 4. Prediksi Normal (Precision 80%, Recall 66.67%)
- ✅ **Action**: Monitoring berkala tiap 3 bulan
- ✅ **Confidence**: Baik
- 📋 **Protocol**: Pertahankan gizi seimbang, imunisasi lengkap

### Rekomendasi Berdasarkan Confidence Score

| Confidence | Action |
|------------|--------|
| **> 85%** | Tindak lanjut sesuai kategori prediksi |
| **70-85%** | Monitoring ketat, re-evaluasi 2 minggu |
| **< 70%** | Pemeriksaan manual komprehensif WAJIB |

---

## 👥 Tim Pengembang

Proyek ini dikembangkan oleh mahasiswa **Informatika, FMIPA, Universitas Syiah Kuala**:

| Nama | NIM | Role |
|------|-----|------|
| **Reyan Andrea** | 2208107010014 | Lead Developer & ML Engineer |
| **Shafa Disya Aulia** | 2308107010002 | UI/UX & Web Development  |
| **Dea Zasqia Pasaribu Malau** | 2308107010004 | Model Analysis & Writer |
| **Tasya Zahrani** | 2308107010006 | Data Analyst & Preprocessing |

**Mata Kuliah**: Kecerdasan Artificial  
**Semester**: Genap 2024/2025  
**Institusi**: Universitas Syiah Kuala, Banda Aceh

---

## 📚 Referensi

1. Kementerian Kesehatan RI. (2023). *Profil Kesehatan Indonesia 2023*. Jakarta: Kemenkes RI.

2. Putri, R., & Ardiansyah, M. (2022). Penerapan Artificial Neural Network untuk Prediksi Gizi Anak Balita. *Jurnal Teknologi Informasi dan Kesehatan*, 10(2), 45–52.

3. Priono, A., et al. (2018). Pengembangan Sistem Cerdas dalam Bidang Kesehatan Masyarakat. *Jurnal Sains dan Aplikasi*, 6(1), 12–19.

4. Ardi, F., Sari, D., & Nugroho, A. (2021). Penerapan Artificial Neural Network untuk Prediksi Penyakit Kronis. *Jurnal Teknologi Kesehatan*, 8(2), 44–51.

5. Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson Education.

6. Rahmawati, N. (2023). Analisis Faktor Sosial Ekonomi terhadap Stunting di Indonesia. *Jurnal Gizi dan Kesehatan*, 11(1), 12–18.

7. WHO. (2021). *Child Growth Standards*. Geneva: World Health Organization.

8. UNICEF. (2020). *State of the World's Children 2020: Children, Food and Nutrition*. New York: UNICEF.

---

## 🚀 Future Development

### Roadmap

- [ ] **Dataset Expansion**: 100 → 1000+ sampel dari berbagai daerah
- [ ] **Feature Enhancement**: Tambah data sosial ekonomi, sanitasi, riwayat kesehatan
- [ ] **Model Improvement**: Hyperparameter tuning, ensemble methods
- [ ] **Recall Optimization**: Target stunting recall 75% → 90%+
- [ ] **System Integration**: API development, mobile app, database Puskesmas
- [ ] **Real-time Monitoring**: Dashboard untuk tracking perkembangan anak
- [ ] **Multi-language Support**: Bahasa Indonesia & English

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dosen Pembimbing Mata Kuliah Kecerdasan Artificial USK
- Kementerian Kesehatan RI untuk data dan referensi stunting
- Dataset: Indonesian Children Medical & Food Nutrition
- TensorFlow & Streamlit Community
- WHO & UNICEF untuk standar pertumbuhan anak

---

## 📞 Contact & Support

### Hubungi Kami

- **GitHub**: [@ReyanAndrea](https://github.com/ReyanAndrea)
- **Repository**: [ANN_StuntingPrediction](https://github.com/ReyanAndrea/ANN_StuntingPrediction)

### Report Issues

Jika menemukan bug atau memiliki saran, silakan [buat issue](https://github.com/ReyanAndrea/ANN_StuntingPrediction/issues) di repository.

### Contribution

Kontribusi sangat welcome! Silakan fork repository dan submit pull request.

---

<div align="center">

### ⭐ Star This Repository ⭐

**Jika proyek ini bermanfaat, jangan lupa berikan star!**

---

**Made with ❤️ by Kelompok 1 - Informatika USK 2025**

![Universitas Syiah Kuala](https://img.shields.io/badge/Universitas-Syiah%20Kuala-blue)
![Informatika](https://img.shields.io/badge/Jurusan-Informatika-green)
![AI Course](https://img.shields.io/badge/MK-Kecerdasan%20Artificial-orange)

</div>
