"""
NutriPredict - Aplikasi Prediksi Stunting
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import joblib
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title="NutriPredict - Prediksi Stunting",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & ARTIFACTS ====================
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = keras.models.load_model('nutripredict_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_names = joblib.load('feature_names.pkl')
        metrics = joblib.load('metrics.pkl')
        return model, scaler, label_encoders, target_encoder, feature_names, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Silakan jalankan training script terlebih dahulu!")
        return None, None, None, None, None, None

model, scaler, label_encoders, target_encoder, feature_names, metrics = load_model_artifacts()

# ==================== SIDEBAR ====================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
st.sidebar.title("🎯 NutriPredict")
st.sidebar.markdown("**Sistem Prediksi Stunting dengan ANN**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Menu Navigasi",
    ["🏠 Home", "🔮 Prediksi Stunting", "📊 Evaluasi Model", "📈 Dashboard Monitoring", "ℹ️ Tentang"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dikembangkan oleh:**")
st.sidebar.markdown("- Reyan Andrea")
st.sidebar.markdown("- Shafa Disya Aulia")
st.sidebar.markdown("- Dea Zasqia Pasaribu Malau")
st.sidebar.markdown("- Tasya Zahrani")
st.sidebar.markdown("\n📍 Universitas Syiah Kuala")

# ==================== FUNGSI REKOMENDASI ====================
def get_recommendations(risk_level, probabilities):
    """Generate personalized intervention recommendations"""
    recommendations = {
        'Rendah': [
            "✅ Pertahankan pola makan bergizi seimbang",
            "✅ Lanjutkan pemantauan pertumbuhan rutin setiap bulan",
            "✅ Pastikan anak mendapat ASI eksklusif (jika < 6 bulan)",
            "✅ Jaga kebersihan lingkungan dan sanitasi rumah",
            "✅ Berikan imunisasi lengkap sesuai jadwal"
        ],
        'Sedang': [
            "⚠️ Tingkatkan asupan protein (telur, ikan, daging, kacang-kacangan)",
            "⚠️ Berikan vitamin dan mineral tambahan (Vitamin A, Zink, Zat Besi)",
            "⚠️ Pemantauan pertumbuhan setiap 2 minggu",
            "⚠️ Perbaiki sanitasi rumah (akses air bersih, toilet sehat)",
            "⚠️ Konsultasi dengan ahli gizi untuk pola makan anak",
            "⚠️ Evaluasi kondisi kesehatan (cek infeksi, cacingan)"
        ],
        'Tinggi': [
            "🚨 SEGERA konsultasi ke dokter/ahli gizi!",
            "🚨 Berikan makanan tambahan khusus (PMT)",
            "🚨 Pemantauan intensif setiap minggu",
            "🚨 Cek kesehatan menyeluruh (laboratorium lengkap)",
            "🚨 Perbaikan sanitasi dan lingkungan PRIORITAS",
            "🚨 Edukasi orang tua tentang pola asuh dan gizi",
            "🚨 Program intervensi gizi terintegrasi",
            "🚨 Follow-up rutin dengan tenaga kesehatan"
        ]
    }
    return recommendations.get(risk_level, [])

# ==================== PAGE: HOME ====================
if menu == "🏠 Home":
    st.markdown("<div class='main-header'>👶 NutriPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Sistem Prediksi Stunting untuk Intervensi Dini berbasis Artificial Neural Network</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("📊 Prevalensi Stunting", "21.5%", "Indonesia 2023")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🎯 Target Pemerintah", "14%", "Tahun 2024")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🧠 Akurasi Model", f"{metrics['accuracy']*100:.2f}%" if metrics else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Apa itu NutriPredict?")
    st.write("""
    **NutriPredict** adalah sistem prediksi stunting berbasis **Artificial Neural Network (ANN)** 
    yang dapat memprediksi risiko stunting pada anak **6-12 bulan ke depan**.
    
    Sistem ini mengintegrasikan berbagai faktor seperti:
    - 🍎 **Status Gizi Anak** (berat, tinggi, ASI eksklusif)
    - 💰 **Kondisi Ekonomi Keluarga**
    - 🚰 **Sanitasi dan Lingkungan**
    - 📚 **Pendidikan Orang Tua**
    """)
    
    st.markdown("### 🚀 Fitur Utama")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔮 Prediksi Risiko**
        - Prediksi tingkat risiko: Rendah, Sedang, Tinggi
        - Skor probabilitas untuk setiap kategori
        - Analisis berbasis data multidimensi
        
        **💡 Rekomendasi Personal**
        - Saran intervensi sesuai tingkat risiko
        - Panduan gizi dan pola asuh
        - Langkah perbaikan sanitasi
        """)
    
    with col2:
        st.markdown("""
        **📊 Dashboard Monitoring**
        - Visualisasi distribusi risiko
        - Tracking efektivitas intervensi
        - Statistik anak berisiko per wilayah
        
        **📈 Evaluasi Model**
        - Confusion Matrix
        - Precision, Recall, F1-Score
        - Analisis performa model
        """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Mulai dengan menu 'Prediksi Stunting' untuk memprediksi risiko stunting anak!")

# ==================== PAGE: PREDIKSI STUNTING ====================
elif menu == "🔮 Prediksi Stunting":
    st.markdown("<div class='main-header'>🔮 Prediksi Risiko Stunting</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model belum tersedia. Silakan jalankan training script terlebih dahulu!")
        st.stop()
    
    st.markdown("### 📝 Input Data Anak")
    st.info("Masukkan data anak untuk memprediksi risiko stunting")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👶 Data Anak")
            age = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=12)
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=75.0, step=0.1)
            weight = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=25.0, value=9.0, step=0.1)
            birth_weight = st.number_input("Berat Lahir (kg)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            breastfeeding = st.selectbox("ASI Eksklusif", ["Ya", "Tidak"])
        
        with col2:
            st.markdown("#### 👨‍👩‍👧 Data Keluarga")
            maternal_education = st.selectbox(
                "Pendidikan Ibu",
                ["SD", "SMP", "SMA", "Diploma/Sarjana"]
            )
            household_income = st.selectbox(
                "Pendapatan Keluarga",
                ["Rendah (< 2 juta)", "Menengah (2-5 juta)", "Tinggi (> 5 juta)"]
            )
            sanitation = st.selectbox(
                "Akses Sanitasi",
                ["Baik (Air bersih + toilet sehat)", "Sedang", "Buruk"]
            )
            water_source = st.selectbox(
                "Sumber Air",
                ["PDAM/Air Kemasan", "Sumur", "Sungai/Tidak Layak"]
            )
        
        submitted = st.form_submit_button("🔮 Prediksi Risiko", use_container_width=True)
    
    if submitted:
        # Prepare input data (disesuaikan dengan feature_names dari training)
        # CATATAN: Ini adalah contoh, sesuaikan dengan kolom dataset asli
        input_data = {
            'Age_months': age,
            'Gender': 1 if gender == "Laki-laki" else 0,
            'Height_cm': height,
            'Weight_kg': weight,
            'Birth_Weight_kg': birth_weight,
            'Breastfeeding': 1 if breastfeeding == "Ya" else 0,
            'Maternal_Education': ["SD", "SMP", "SMA", "Diploma/Sarjana"].index(maternal_education),
            'Household_Income': ["Rendah (< 2 juta)", "Menengah (2-5 juta)", "Tinggi (> 5 juta)"].index(household_income),
            'Sanitation': ["Buruk", "Sedang", "Baik (Air bersih + toilet sehat)"].index(sanitation),
            'Water_Source': ["Sungai/Tidak Layak", "Sumur", "PDAM/Air Kemasan"].index(water_source)
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features from training are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value untuk missing features
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction_proba = model.predict(input_scaled, verbose=0)[0]
        prediction_class = np.argmax(prediction_proba)
        risk_level = target_encoder.inverse_transform([prediction_class])[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")
        
        # Risk Level Card
        risk_colors = {
            'Rendah': 'risk-low',
            'Sedang': 'risk-medium',
            'Tinggi': 'risk-high'
        }
        risk_icons = {
            'Rendah': '✅',
            'Sedang': '⚠️',
            'Tinggi': '🚨'
        }
        
        st.markdown(f"<div class='{risk_colors.get(risk_level, 'risk-low')}'>", unsafe_allow_html=True)
        st.markdown(f"## {risk_icons.get(risk_level, '❓')} Tingkat Risiko: **{risk_level.upper()}**")
        st.markdown(f"**Probabilitas:** {prediction_proba[prediction_class]*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Probability Distribution
        st.markdown("#### 📊 Distribusi Probabilitas")
        prob_df = pd.DataFrame({
            'Kategori': target_encoder.classes_,
            'Probabilitas': prediction_proba * 100
        })
        
        fig = px.bar(
            prob_df,
            x='Kategori',
            y='Probabilitas',
            color='Kategori',
            color_discrete_map={'Rendah': '#28a745', 'Sedang': '#ffc107', 'Tinggi': '#dc3545'},
            text='Probabilitas'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Rekomendasi Intervensi")
        recommendations = get_recommendations(risk_level, prediction_proba)
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Download Report
        st.markdown("---")
        report = f"""
LAPORAN PREDIKSI STUNTING - NutriPredict
==========================================

DATA ANAK:
- Umur: {age} bulan
- Jenis Kelamin: {gender}
- Tinggi Badan: {height} cm
- Berat Badan: {weight} kg
- Berat Lahir: {birth_weight} kg
- ASI Eksklusif: {breastfeeding}

DATA KELUARGA:
- Pendidikan Ibu: {maternal_education}
- Pendapatan: {household_income}
- Sanitasi: {sanitation}
- Sumber Air: {water_source}

HASIL PREDIKSI:
- Tingkat Risiko: {risk_level.upper()}
- Probabilitas: {prediction_proba[prediction_class]*100:.2f}%

REKOMENDASI INTERVENSI:
{chr(10).join(['- ' + rec for rec in recommendations])}

==========================================
Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📥 Download Laporan",
            data=report,
            file_name=f"laporan_prediksi_stunting_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ==================== PAGE: EVALUASI MODEL ====================
elif menu == "📊 Evaluasi Model":
    st.markdown("<div class='main-header'>📊 Evaluasi Model ANN</div>", unsafe_allow_html=True)
    
    if metrics is None:
        st.error("❌ Metrics belum tersedia. Silakan jalankan training script terlebih dahulu!")
        st.stop()
    
    # Overall Metrics
    st.markdown("### 🎯 Performa Model")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with col2:
        st.metric("Avg Precision", f"{np.mean(metrics['precision']):.4f}")
    with col3:
        st.metric("Avg Recall", f"{np.mean(metrics['recall']):.4f}")
    with col4:
        st.metric("Avg F1-Score", f"{np.mean(metrics['f1_score']):.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    cm = np.array(metrics['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=metrics['target_names'],
        yticklabels=metrics['target_names'],
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    st.pyplot(fig)
    
    # Per-Class Metrics
    st.markdown("### 📈 Metrik Per Kelas")
    
    metrics_df = pd.DataFrame({
        'Kelas': metrics['target_names'],
        'Precision': [f"{p:.4f}" for p in metrics['precision']],
        'Recall': [f"{r:.4f}" for r in metrics['recall']],
        'F1-Score': [f"{f:.4f}" for f in metrics['f1_score']]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # False Positives & False Negatives Analysis
    st.markdown("### ⚠️ Analisis False Positives & False Negatives")
    
    for i, class_name in enumerate(metrics['target_names']):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        with st.expander(f"📊 {class_name.upper()}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("True Positives", tp)
            col2.metric("False Positives", fp)
            col3.metric("False Negatives", fn)
            
            if class_name == 'Tinggi' and fn > 0:
                st.error(f"⚠️ CRITICAL: {fn} anak berisiko TINGGI diprediksi lebih rendah! Mereka butuh intervensi segera.")
    
    # Training History (if available)
    try:
        st.markdown("---")
        st.markdown("### 📈 Training History")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                img = Image.open('training_history.png')
                st.image(img, caption='Training & Validation Metrics')
            except:
                st.info("Training history plot tidak tersedia")
        
        with col2:
            try:
                img = Image.open('class_distribution.png')
                st.image(img, caption='Class Distribution (Before & After SMOTE)')
            except:
                st.info("Class distribution plot tidak tersedia")
    except:
        pass

# ==================== PAGE: DASHBOARD MONITORING ====================
elif menu == "📈 Dashboard Monitoring":
    st.markdown("<div class='main-header'>📈 Dashboard Monitoring</div>", unsafe_allow_html=True)
    
    # Generate dummy data for demo
    np.random.seed(42)
    n_samples = 500
    
    demo_data = pd.DataFrame({
        'Nama': [f"Anak-{i+1}" for i in range(n_samples)],
        'Umur': np.random.randint(0, 60, n_samples),
        'Risiko': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples, p=[0.5, 0.3, 0.2]),
        'Wilayah': np.random.choice(['Aceh Besar', 'Banda Aceh', 'Aceh Utara', 'Aceh Barat', 'Pidie'], n_samples),
        'Status_Gizi': np.random.choice(['Normal', 'Gizi Kurang', 'Gizi Buruk'], n_samples, p=[0.6, 0.25, 0.15])
    })
    
    # Summary Metrics
    st.markdown("### 📊 Ringkasan Statistik")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anak", len(demo_data))
    with col2:
        st.metric("Risiko Tinggi", len(demo_data[demo_data['Risiko'] == 'Tinggi']))
    with col3:
        st.metric("Risiko Sedang", len(demo_data[demo_data['Risiko'] == 'Sedang']))
    with col4:
        st.metric("Risiko Rendah", len(demo_data[demo_data['Risiko'] == 'Rendah']))
    
    st.markdown("---")
    
    # Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Distribusi Tingkat Risiko")
        risk_counts = demo_data['Risiko'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Rendah': '#28a745', 'Sedang': '#ffc107', 'Tinggi': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📍 Risiko per Wilayah")
        wilayah_risk = pd.crosstab(demo_data['Wilayah'], demo_data['Risiko'])
        fig = px.bar(
            wilayah_risk,
            barmode='group',
            color_discrete_map={'Rendah': '#28a745', 'Sedang': '#ffc107', 'Tinggi': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.markdown("---")
    st.markdown("### 📋 Data Anak Berisiko Tinggi")
    high_risk = demo_data[demo_data['Risiko'] == 'Tinggi'].head(20)
    st.dataframe(high_risk, use_container_width=True)
    
    # Download
    csv = demo_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data Lengkap (CSV)",
        data=csv,
        file_name="monitoring_stunting.csv",
        mime="text/csv"
    )

# ==================== PAGE: TENTANG ====================
elif menu == "ℹ️ Tentang":
    st.markdown("<div class='main-header'>ℹ️ Tentang NutriPredict</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Tujuan Sistem
    
    **NutriPredict** dikembangkan untuk:
    1. Memprediksi risiko stunting 6-12 bulan ke depan
    2. Memberikan rekomendasi intervensi dini yang personal
    3. Membantu tenaga kesehatan melakukan deteksi dini lebih efektif
    4. Mendukung program nasional penurunan stunting hingga target 14%
    
    ### 🧠 Teknologi
    
    - **Model:** Artificial Neural Network (ANN) / Multilayer Perceptron (MLP)
    - **Framework:** TensorFlow/Keras
    - **Handling Imbalance:** SMOTE + Class Weighting
    - **Evaluasi:** Confusion Matrix, Precision, Recall, F1-Score
    - **Frontend:** Streamlit
    
    ### 📊 Dataset
    
    Dataset yang digunakan: **Stunting Dataset Indonesia** dari Kaggle
    - Berisi indikator gizi, sosial ekonomi, dan kesehatan anak
    - Data dari berbagai wilayah di Indonesia
    
    ### 👥 Tim Pengembang
    
    Sistem ini dikembangkan oleh mahasiswa Informatika Universitas Syiah Kuala:
    - Reyan Andrea (2208107010014)
    - Shafa Disya Aulia (2308107010002)
    - Dea Zasqia Pasaribu Malau (2308107010004)
    - Tasya Zahrani (2308107010006)
    
    ### 📚 Referensi
    
    1. Kementerian Kesehatan RI. (2023). Profil Kesehatan Indonesia 2023
    2. WHO. (2021). Child Growth Standards
    3. UNICEF. (2020). State of the World's Children 2020
    4. Putri & Ardiansyah. (2022). Penerapan ANN untuk Prediksi Gizi Anak Balita
    
    ### 📧 Kontak
    
    Untuk informasi lebih lanjut atau feedback, silakan hubungi tim pengembang melalui email institusi.
    
    ---
    
    © 2025 NutriPredict - Universitas Syiah Kuala
    """)
    
    st.success("💡 Sistem ini dikembangkan sebagai bagian dari Project Akhir Mata Kuliah Kecerdasan Artificial")