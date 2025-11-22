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
    [data-testid="stSidebar"] [data-testid="image"] {
        display: flex;
        justify-content: center;
        margin: 0 auto;
        padding: 0 20px;
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
st.sidebar.image("logo usk.png", width=100)
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
def get_recommendations(status_gizi, probabilities):
    """Generate personalized nutrition recommendations based on nutritional status"""
    recommendations = {
        'Normal': [
            "✅ Status gizi anak dalam kategori normal",
            "✅ Pertahankan pola makan bergizi seimbang",
            "✅ Terus pantau pertumbuhan secara rutin setiap bulan",
            "✅ Pastikan anak mendapat cukup protein, karbohidrat, lemak, vitamin & mineral",
            "✅ Jaga kebersihan makanan dan sanitasi lingkungan",
            "✅ Lengkapi imunisasi sesuai jadwal"
        ],
        'Overweight': [
            "⚠️ Status gizi anak dalam kategori overweight (berat badan berlebih)",
            "⚠️ Kurangi asupan kalori berlebih (terutama makanan tinggi gula & lemak)",
            "⚠️ Tingkatkan aktivitas fisik dan bermain anak",
            "⚠️ Perbanyak konsumsi sayur dan buah-buahan",
            "⚠️ Batasi konsumsi minuman manis dan makanan cepat saji",
            "⚠️ Konsultasikan dengan ahli gizi untuk pola makan sehat"
        ],
        'Underweight': [
            "🚨 Status gizi anak dalam kategori kurang gizi (underweight)",
            "🚨 SEGERA tingkatkan asupan nutrisi anak",
            "🚨 Berikan makanan tinggi kalori & protein (telur, ikan, daging, kacang, susu)",
            "🚨 Berikan vitamin & mineral tambahan (Vitamin A, Zink, Zat Besi)",
            "🚨 Pemantauan pertumbuhan setiap 2 minggu",
            "🚨 Konsultasi dengan dokter/ahli gizi untuk intervensi gizi lanjutan",
            "🚨 Perbaiki sanitasi rumah (akses air bersih, toilet sehat)",
            "🚨 Cek kondisi kesehatan menyeluruh (ada infeksi atau cacingan?)"
        ],
        'Stunting': [
            "🚨 STATUS KRITIS: Anak mengalami stunting (gizi buruk kronis)",
            "🚨 SEGERA konsultasi ke dokter atau pusat kesehatan!",
            "🚨 Berikan makanan tambahan khusus & program makanan bergizi (PMT)",
            "🚨 Pemantauan intensif setiap minggu dengan tenaga kesehatan",
            "🚨 Cek kesehatan menyeluruh (lab lengkap, cari infeksi penyebab)",
            "🚨 Edukasi orang tua tentang pola asuh dan gizi anak",
            "🚨 Perbaikan sanitasi & lingkungan rumah PRIORITAS UTAMA",
            "🚨 Program intervensi gizi terintegrasi dengan dukungan pemerintah/desa"
        ]
    }
    return recommendations.get(status_gizi, [])

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
    **NutriPredict** adalah sistem prediksi status gizi berbasis **Artificial Neural Network (ANN)** 
    yang dapat memprediksi status gizi anak berdasarkan data antropometri (ukuran tubuh).
    
    Sistem ini dapat mengklasifikasikan anak ke dalam 4 kategori status gizi:
    - 🟢 **Normal** - Status gizi sesuai standar WHO
    - 🟡 **Overweight** - Berat badan berlebih
    - 🔴 **Underweight** - Gizi kurang (berat badan kurang)
    - 🔴 **Stunting** - Gizi buruk kronis (tinggi badan kurang)
    
    Data yang digunakan untuk prediksi:
    - 👶 **Umur Anak** (bulan)
    - 👫 **Jenis Kelamin**
    - 📏 **Tinggi Badan** (cm)
    - ⚖️ **Berat Badan** (kg)
    """)
    
    st.markdown("### 🚀 Fitur Utama")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔮 Prediksi Status Gizi**
        - Klasifikasi ke 4 kategori: Normal, Overweight, Underweight, Stunting
        - Skor probabilitas untuk setiap kategori
        - Analisis berbasis data antropometri
        
        **💡 Rekomendasi Personal**
        - Saran nutrisi sesuai status gizi
        - Panduan intervensi gizi yang tepat
        - Langkah monitoring kesehatan anak
        """)
    
    with col2:
        st.markdown("""
        **📊 Dashboard Monitoring**
        - Visualisasi distribusi status gizi
        - Tracking efektivitas program
        - Statistik per wilayah/daerah
        
        **📈 Evaluasi Model**
        - Confusion Matrix
        - Precision, Recall, F1-Score
        - Analisis performa klasifikasi
        """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Mulai dengan menu 'Prediksi Stunting' untuk memprediksi status gizi anak!")

# ==================== PAGE: PREDIKSI STUNTING ====================
elif menu == "🔮 Prediksi Stunting":
    st.markdown("<div class='main-header'>🔮 Prediksi Risiko Stunting</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model belum tersedia. Silakan jalankan training script terlebih dahulu!")
        st.stop()
    
    st.markdown("### 📝 Input Data Anak")
    st.info("Masukkan data anak untuk memprediksi status gizi/risiko stunting")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👶 Data Anak")
            age = st.number_input("Umur (bulan)", min_value=0, max_value=120, value=12)
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=150.0, value=75.0, step=0.1)
            weight = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0, value=9.0, step=0.1)
        
        with col2:
            st.markdown("#### 📊 Info Tambahan")
            st.info("Pastikan nilai Age, Height, dan Weight sudah benar sebelum prediksi")
        
        submitted = st.form_submit_button("🔮 Prediksi Status Gizi", use_container_width=True)
    
    if submitted:
        try:
            # Validate input
            if age < 0 or age > 120:
                st.error("❌ Umur harus antara 0-120 bulan")
                st.stop()
            if height < 40 or height > 150:
                st.error("❌ Tinggi badan harus antara 40-150 cm")
                st.stop()
            if weight < 1 or weight > 30:
                st.error("❌ Berat badan harus antara 1-30 kg")
                st.stop()
            
            # Prepare input data sesuai dengan feature_names dari training
            input_data = {
                'Sex': 1 if gender == "Male" else 0,
                'Age': age,
                'Height': height,
                'Weight': weight
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features from training are present dengan order yang benar
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
            
            # Status Card with color coding
            status_colors = {
                'Underweight': 'risk-high',
                'Stunting': 'risk-high',
                'Normal': 'risk-low',
                'Overweight': 'risk-medium'
            }
            status_icons = {
                'Underweight': '🚨',
                'Stunting': '🚨',
                'Normal': '✅',
                'Overweight': '⚠️'
            }
            
            st.markdown(f"<div class='{status_colors.get(risk_level, 'risk-low')}'>", unsafe_allow_html=True)
            st.markdown(f"## {status_icons.get(risk_level, '❓')} Status Gizi: **{risk_level.upper()}**")
            st.markdown(f"**Probabilitas:** {prediction_proba[prediction_class]*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Probability Distribution
            st.markdown("#### 📊 Distribusi Probabilitas Semua Kategori")
            prob_df = pd.DataFrame({
                'Status': target_encoder.classes_,
                'Probabilitas': prediction_proba * 100
            })
            
            fig = px.bar(
                prob_df,
                x='Status',
                y='Probabilitas',
                color='Status',
                color_discrete_map={
                    'Normal': '#28a745',
                    'Overweight': '#ffc107',
                    'Underweight': '#dc3545',
                    'Stunting': '#c82333'
                },
                text='Probabilitas'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show input data summary
            st.markdown("#### 📋 Data Input yang Digunakan")
            input_summary = pd.DataFrame({
                'Parameter': ['Umur', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan'],
                'Nilai': [f'{age} bulan', gender, f'{height} cm', f'{weight} kg']
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
            
            # Recommendations berdasarkan status gizi
            st.markdown("### 💡 Rekomendasi & Penjelasan")
            recommendations = get_recommendations(risk_level, prediction_proba)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Download Report
            st.markdown("---")
            report = f"""
LAPORAN PREDIKSI STATUS GIZI - NutriPredict
============================================

DATA ANAK:
- Umur: {age} bulan
- Jenis Kelamin: {gender}
- Tinggi Badan: {height} cm
- Berat Badan: {weight} kg

HASIL PREDIKSI:
- Status Gizi: {risk_level.upper()}
- Probabilitas: {prediction_proba[prediction_class]*100:.2f}%

DISTRIBUSI PROBABILITAS:
"""
            for i, class_name in enumerate(target_encoder.classes_):
                report += f"- {class_name}: {prediction_proba[i]*100:.2f}%\n"
            
            report += f"""
REKOMENDASI INTERVENSI:
{chr(10).join(['- ' + rec for rec in recommendations])}

============================================
Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Sistem: NutriPredict ANN
            """
            
            st.download_button(
                label="📥 Download Laporan",
                data=report,
                file_name=f"laporan_prediksi_status_gizi_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
            st.info("Pastikan model sudah dilatih dengan benar. Jalankan train_model.py terlebih dahulu.")

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
            
            if class_name in ['Stunting', 'Underweight'] and fn > 0:
                st.error(f"⚠️ CRITICAL: {fn} anak dengan status '{class_name}' diprediksi salah! Mereka butuh perhatian khusus.")
    
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
    
    try:
        df_kemenkes = pd.read_csv('data-stunting2022.csv')

        if 'tahun' in df_kemenkes.columns:
            df_kemenkes = df_kemenkes[df_kemenkes['tahun'] == 2022]

        df_kemenkes = df_kemenkes.rename(columns={
            'kemendagri_nama_kabupaten_kota': 'Wilayah',
            'jenis_stunting': 'Jenis',
            'jumlah': 'Jumlah',
            'tahun': 'Tahun',
            'bulan': 'Bulan'
        })

        df_kemenkes['Status_Gizi'] = df_kemenkes['Jenis'].astype(str)

        def jenis_to_risiko(j):
            s = str(j).lower()
            if 'stunt' in s or 'stunting' in s:
                return 'Tinggi'
            if 'under' in s or 'underweight' in s or 'kurang' in s:
                return 'Tinggi'
            if 'over' in s or 'overweight' in s or 'lebih' in s:
                return 'Sedang'
            return 'Sedang'

        df_kemenkes['Risiko'] = df_kemenkes['Jenis'].apply(jenis_to_risiko)

        demo_data = df_kemenkes[['Wilayah', 'Risiko', 'Status_Gizi', 'Jumlah', 'Bulan', 'Tahun']].copy()
        demo_data = demo_data.rename(columns={'Wilayah': 'Nama'})

    except Exception:
        try:
            df_raw = pd.read_csv('child_data_rev.csv')
            df_raw = df_raw.rename(columns={
                'Name': 'Nama', 'Age': 'Umur', 'Status': 'Status_Gizi'
            })
            if 'Wilayah' not in df_raw.columns:
                df_raw['Wilayah'] = 'Unknown'
            def map_risiko(status):
                s = str(status).lower()
                if 'stunt' in s or 'under' in s or 'kurang' in s:
                    return 'Tinggi'
                if 'over' in s or 'lebih' in s:
                    return 'Sedang'
                if 'normal' in s:
                    return 'Rendah'
                return 'Sedang'
            df_raw['Risiko'] = df_raw['Status_Gizi'].apply(map_risiko)
            demo_data = df_raw.rename(columns={'Umur': 'Umur (bulan)', 'Status_Gizi': 'Status_Gizi'})
        except Exception:
            np.random.seed(42)
            n_samples = 500
            demo_data = pd.DataFrame({
                'Nama': [f"Anak-{i+1}" for i in range(n_samples)],
                'Umur (bulan)': np.random.randint(0, 60, n_samples),
                'Risiko': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples, p=[0.5, 0.3, 0.2]),
                'Wilayah': np.random.choice(['Aceh Besar', 'Banda Aceh', 'Aceh Utara', 'Aceh Barat', 'Pidie'], n_samples),
                'Status_Gizi': np.random.choice(['Normal', 'Gizi Kurang', 'Gizi Buruk'], n_samples, p=[0.6, 0.25, 0.15])
            })
    
    st.markdown("### 📊 Ringkasan Statistik Stunting di Aceh per tahun 2022")
    col1, col2, col3, col4 = st.columns(4)

    if 'Jumlah' in demo_data.columns:
        total_count = int(demo_data['Jumlah'].sum())
        tinggi_count = int(demo_data.loc[demo_data['Risiko'] == 'Tinggi', 'Jumlah'].sum())
        sedang_count = int(demo_data.loc[demo_data['Risiko'] == 'Sedang', 'Jumlah'].sum())
        rendah_count = int(demo_data.loc[demo_data['Risiko'] == 'Rendah', 'Jumlah'].sum())
    else:
        total_count = len(demo_data)
        tinggi_count = len(demo_data[demo_data['Risiko'] == 'Tinggi'])
        sedang_count = len(demo_data[demo_data['Risiko'] == 'Sedang'])
        rendah_count = len(demo_data[demo_data['Risiko'] == 'Rendah'])

    with col1:
        st.metric("Total Anak / Kasus", total_count)
    with col2:
        st.metric("Risiko Tinggi", tinggi_count)
    with col3:
        st.metric("Risiko Sedang", sedang_count)
    with col4:
        st.metric("Risiko Rendah", rendah_count)

    st.markdown("---")

    # Distribution Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Distribusi Tingkat Risiko")
        if 'Jumlah' in demo_data.columns:
            risk_counts = demo_data.groupby('Risiko')['Jumlah'].sum().reindex(['Rendah', 'Sedang', 'Tinggi']).fillna(0)
        else:
            risk_counts = demo_data['Risiko'].value_counts().reindex(['Rendah', 'Sedang', 'Tinggi']).fillna(0)

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Rendah': '#28a745', 'Sedang': '#ffc107', 'Tinggi': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📍 Risiko per Wilayah")
        region_col = 'Wilayah' if 'Wilayah' in demo_data.columns else 'Nama'
        if 'Jumlah' in demo_data.columns:
            wilayah_risk = demo_data.pivot_table(index=region_col, columns='Risiko', values='Jumlah', aggfunc='sum').fillna(0)
        else:
            wilayah_risk = pd.crosstab(demo_data[region_col], demo_data['Risiko'])

        fig = px.bar(
            wilayah_risk,
            barmode='group',
            color_discrete_map={'Rendah': '#28a745', 'Sedang': '#ffc107', 'Tinggi': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data Table
    st.markdown("---")
    st.markdown("### 📋 Data Anak / Daerah Berisiko Tinggi")
    if 'Jumlah' in demo_data.columns:
        high_risk = demo_data[demo_data['Risiko'] == 'Tinggi'].sort_values('Jumlah', ascending=False).head(20)
    else:
        high_risk = demo_data[demo_data['Risiko'] == 'Tinggi'].head(20)

    # Angka di sebelah nama wilayah adalah indeks baris (pandas DataFrame index) —
    # biasanya muncul karena kita memfilter/menyortir sehingga index asli tetap.
    # Reset index (tanpa menambahkan kolom baru) dan sembunyikan index di tampilan Streamlit.
    st.dataframe(high_risk.reset_index(drop=True), use_container_width=True, hide_index=True)

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
    1. Memprediksi status gizi anak berdasarkan data antropometri
    2. Memberikan rekomendasi intervensi gizi yang personal dan tepat sasaran
    3. Membantu tenaga kesehatan melakukan deteksi dini malnutrisi
    4. Mendukung program nasional penurunan stunting hingga target 14%
    
    ### 🧠 Teknologi & Metodologi
    
    - **Model:** Artificial Neural Network (ANN) / Multilayer Perceptron (MLP)
    - **Framework:** TensorFlow/Keras
    - **Arsitektur:** 
      - Input Layer (4 fitur: Age, Sex, Height, Weight)
      - Hidden Layer 1 (128 neuron, ReLU, Dropout 0.3)
      - Hidden Layer 2 (64 neuron, ReLU, Dropout 0.3)
      - Hidden Layer 3 (32 neuron, ReLU, Dropout 0.2)
      - Output Layer (4 neuron, Softmax untuk klasifikasi multi-kelas)
    - **Handling Imbalance:** SMOTE + Class Weighting
    - **Evaluasi:** Confusion Matrix, Precision, Recall, F1-Score
    - **Frontend:** Streamlit
    
    ### 📊 Dataset
    
    Dataset yang digunakan: **Child Growth and Nutrition Dataset** dari Kaggle
    - Berisi data antropometri anak (Age, Height, Weight)
    - Jenis Kelamin dan Status Gizi dari berbagai negara
    - Total data: 1000+ records
    - Pembagian data: 80% training, 20% testing
    
    ### 👥 Tim Pengembang
    
    Sistem ini dikembangkan oleh mahasiswa Informatika Universitas Syiah Kuala:
    - Reyan Andrea (2208107010014)
    - Shafa Disya Aulia (2308107010002)
    - Dea Zasqia Pasaribu Malau (2308107010004)
    - Tasya Zahrani (2308107010006)
    
    ### 📚 Referensi
    
    1. WHO. (2021). Child Growth Standards and Anthropometric Indices
    2. UNICEF. (2023). Global Nutrition Report
    3. Kementerian Kesehatan RI. (2023). Profil Kesehatan Indonesia
    4. Roesli et al. (2021). Artificial Neural Networks untuk Klasifikasi Status Gizi Anak
    
    ---
    
    © 2025 NutriPredict - kelompok 1 - Project UAS mata kuliah Kecerdasan Artificial - Universitas Syiah Kuala - Semua Hak Dilindungi
    """)
    
    st.success("💡 Sistem ini dikembangkan sebagai bagian dari Project Akhir Mata Kuliah Kecerdasan Artificial - Prediksi Status Gizi Anak dengan Artificial Neural Network")

