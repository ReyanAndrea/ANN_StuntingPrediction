"""
NutriPredict - Aplikasi Prediksi Stunting
Streamlit Web Application - Modern UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as pltpython 
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

# Modern Custom CSS with Futuristic Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header Styles */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Card Styles with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Risk Level Cards */
    .risk-card {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 2px solid;
        animation: slideInRight 0.6s ease-out;
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-color: #10b981;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-color: #fbbf24;
        box-shadow: 0 10px 30px rgba(251, 191, 36, 0.2);
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border-color: #ef4444;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    }
    
    /* Feature Card */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Sidebar Title Styling */
    [data-testid="stSidebar"] h2 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar Subtitle */
    [data-testid="stSidebar"] p {
        font-size: 0.9rem !important;
        opacity: 0.95;
        line-height: 1.4;
    }
    
    /* Radio Button Container - Remove Background */
    [data-testid="stSidebar"] .stRadio {
        background: transparent;
        padding: 0;
        margin: 1.5rem 0;
    }
    
    /* Hide Radio Button Circles */
    [data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > div > div:first-child {
        display: none !important;
    }
    
    /* Individual Radio Options - Clean Modern Style */
    [data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        padding: 0.85rem 1.2rem !important;
        border-radius: 10px !important;
        margin: 0.4rem 0 !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: block !important;
        border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
        transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Selected Radio Button - Active State */
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
        border-color: white !important;
    }
    
    /* Team Section Styling */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25) !important;
        margin: 2rem 0 1.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] h5 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        opacity: 0.9;
    }
    
    /* Team Members List */
    [data-testid="stSidebar"] ul {
        list-style: none !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] li {
        padding: 0.3rem 0 !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Info/Success/Warning/Error Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Data Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* Logo centering and alignment */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin: 1rem auto 1.5rem auto;
        padding: 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        display: block;
        margin: 0 auto;
    }
    
    /* Form Container */
    [data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
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
with st.sidebar:
    st.image("logo usk putih.png", width=120)
    st.markdown("## ⚡ NutriPredict")
    st.markdown("**Sistem Prediksi Stunting dengan AI**")
    st.markdown("---")

    # Custom navigation with icons
    st.markdown("""
        <style>
        .nav-item {
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        .nav-icon {
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Menu Navigasi",
        ["Home", "Prediksi Stunting", "Evaluasi Model", "Dashboard Monitoring", "Tentang"],
        format_func=lambda x: {
            "Home": "🏡 Home",
            "Prediksi Stunting": "🔬 Prediksi Stunting", 
            "Evaluasi Model": "📉 Evaluasi Model",
            "Dashboard Monitoring": "📊 Dashboard Monitoring",
            "Tentang": "ℹ️ Tentang"
        }[x],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### 👥 Tim Pengembang")
    st.markdown("""
    • Reyan Andrea  
      (2208107010014)
    
    • Shafa Disya Aulia  
      (2308107010002)
    
    • Dea Zasqia Pasaribu Malau  
      (2308107010004)
    
    • Tasya Zahrani  
      (2308107010006)
    """)
    st.markdown("\n📍 **Universitas Syiah Kuala**")

# ==================== FUNGSI REKOMENDASI ====================
def get_recommendations(status_gizi, probabilities):
    """Generate personalized nutrition recommendations based on nutritional status"""
    recommendations = {
        'Normal': [
            "✓ Status gizi anak dalam kategori normal",
            "✓ Pertahankan pola makan bergizi seimbang",
            "✓ Terus pantau pertumbuhan secara rutin setiap bulan",
            "✓ Pastikan anak mendapat cukup protein, karbohidrat, lemak, vitamin & mineral",
            "✓ Jaga kebersihan makanan dan sanitasi lingkungan",
            "✓ Lengkapi imunisasi sesuai jadwal"
        ],
        'Overweight': [
            "▲ Status gizi anak dalam kategori overweight (berat badan berlebih)",
            "▲ Kurangi asupan kalori berlebih (terutama makanan tinggi gula & lemak)",
            "▲ Tingkatkan aktivitas fisik dan bermain anak",
            "▲ Perbanyak konsumsi sayur dan buah-buahan",
            "▲ Batasi konsumsi minuman manis dan makanan cepat saji",
            "▲ Konsultasikan dengan ahli gizi untuk pola makan sehat"
        ],
        'Underweight': [
            "⚠ Status gizi anak dalam kategori kurang gizi (underweight)",
            "⚠ SEGERA tingkatkan asupan nutrisi anak",
            "⚠ Berikan makanan tinggi kalori & protein (telur, ikan, daging, kacang, susu)",
            "⚠ Berikan vitamin & mineral tambahan (Vitamin A, Zink, Zat Besi)",
            "⚠ Pemantauan pertumbuhan setiap 2 minggu",
            "⚠ Konsultasi dengan dokter/ahli gizi untuk intervensi gizi lanjutan",
            "⚠ Perbaiki sanitasi rumah (akses air bersih, toilet sehat)",
            "⚠ Cek kondisi kesehatan menyeluruh (ada infeksi atau cacingan?)"
        ],
        'Stunting': [
            "⚠ STATUS KRITIS: Anak mengalami stunting (gizi buruk kronis)",
            "⚠ SEGERA konsultasi ke dokter atau pusat kesehatan!",
            "⚠ Berikan makanan tambahan khusus & program makanan bergizi (PMT)",
            "⚠ Pemantauan intensif setiap minggu dengan tenaga kesehatan",
            "⚠ Cek kesehatan menyeluruh (lab lengkap, cari infeksi penyebab)",
            "⚠ Edukasi orang tua tentang pola asuh dan gizi anak",
            "⚠ Perbaikan sanitasi & lingkungan rumah PRIORITAS UTAMA",
            "⚠ Program intervensi gizi terintegrasi dengan dukungan pemerintah/desa"
        ]
    }
    return recommendations.get(status_gizi, [])

# ==================== PAGE: HOME ====================
if menu == "Home":
    st.markdown("<div class='main-header'>⚡ NutriPredict</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Sistem Prediksi Stunting untuk Intervensi Dini berbasis Artificial Intelligence</div>", unsafe_allow_html=True)
    
    # Metrics dengan card design yang lebih modern
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='color: #667eea; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                📊 Prevalensi Stunting
            </div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.3rem;'>
                21.5%
            </div>
            <div style='color: #ef4444; font-size: 0.9rem; font-weight: 500;'>
                ↑ Indonesia 2023
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='color: #667eea; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                🎯 Target Pemerintah
            </div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.3rem;'>
                14%
            </div>
            <div style='color: #10b981; font-size: 0.9rem; font-weight: 500;'>
                ↓ Tahun 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='color: #667eea; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                ⚡ Akurasi Model AI
            </div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.3rem;'>
                """ + (f"{metrics['accuracy']*100:.2f}%" if metrics else "N/A") + """
            </div>
            <div style='color: #667eea; font-size: 0.9rem; font-weight: 500;'>
                Neural Network
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # About Section dengan design lebih menarik
    st.markdown("<div class='section-header'>▸ Tentang NutriPredict</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #334155;'>
    <strong>NutriPredict</strong> adalah sistem prediksi status gizi berbasis <strong>Artificial Intelligence (AI)</strong> 
    yang menggunakan teknologi <strong>Artificial Neural Network (ANN)</strong> untuk memprediksi status gizi anak 
    berdasarkan data antropometri (ukuran tubuh).
    </p>
    
    <div style='margin-top: 1.5rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 12px; border-left: 4px solid #667eea;'>
    <h4 style='color: #667eea; margin-bottom: 1rem;'>▸ Kategori Status Gizi:</h4>
    <ul style='list-style: none; padding: 0;'>
        <li style='padding: 0.5rem 0;'><span style='color: #10b981; font-size: 1.3rem; font-weight: bold;'>●</span> <strong>Normal</strong> - Status gizi sesuai standar WHO</li>
        <li style='padding: 0.5rem 0;'><span style='color: #fbbf24; font-size: 1.3rem; font-weight: bold;'>●</span> <strong>Overweight</strong> - Berat badan berlebih</li>
        <li style='padding: 0.5rem 0;'><span style='color: #f59e0b; font-size: 1.3rem; font-weight: bold;'>●</span> <strong>Underweight</strong> - Gizi kurang</li>
        <li style='padding: 0.5rem 0;'><span style='color: #ef4444; font-size: 1.3rem; font-weight: bold;'>●</span> <strong>Stunting</strong> - Gizi buruk kronis</li>
    </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("<div class='section-header'>✦ Fitur Unggulan</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>◆ Prediksi Akurat</h3>
            <ul style='line-height: 2;'>
                <li>Klasifikasi 4 kategori status gizi</li>
                <li>Skor probabilitas real-time</li>
                <li>Analisis berbasis AI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>◆ Dashboard Lengkap</h3>
            <ul style='line-height: 2;'>
                <li>Visualisasi data interaktif</li>
                <li>Tracking program kesehatan</li>
                <li>Statistik per wilayah</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>◆ Rekomendasi Personal</h3>
            <ul style='line-height: 2;'>
                <li>Saran nutrisi spesifik</li>
                <li>Panduan intervensi tepat</li>
                <li>Monitoring kesehatan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>◆ Evaluasi Model</h3>
            <ul style='line-height: 2;'>
                <li>Confusion Matrix detail</li>
                <li>Metrik performa lengkap</li>
                <li>Analisis akurasi AI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 20px; text-align: center; color: white;'>
        <h2 style='color: white; margin-bottom: 1rem;'>⚡ Siap Memulai?</h2>
        <p style='font-size: 1.2rem; margin-bottom: 2rem;'>Gunakan menu 'Prediksi Stunting' untuk memprediksi status gizi anak sekarang!</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: PREDIKSI STUNTING ====================
elif menu == "Prediksi Stunting":
    st.markdown("<div class='main-header'>⚡ Prediksi Status Gizi Anak</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Masukkan data antropometri anak untuk prediksi berbasis AI</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model belum tersedia. Silakan jalankan training script terlebih dahulu!")
        st.stop()
    
    # Create form with modern design
    with st.form("prediction_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### ▸ Data Antropometri Anak")
            
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                age = st.number_input("◆ Umur (bulan)", min_value=0, max_value=120, value=12, help="Masukkan umur anak dalam bulan")
                height = st.number_input("◆ Tinggi Badan (cm)", min_value=40.0, max_value=150.0, value=75.0, step=0.1, help="Masukkan tinggi badan dalam cm")
            
            with col_input2:
                gender = st.selectbox("◆ Jenis Kelamin", ["Male", "Female"], help="Pilih jenis kelamin anak")
                weight = st.number_input("◆ Berat Badan (kg)", min_value=1.0, max_value=30.0, value=9.0, step=0.1, help="Masukkan berat badan dalam kg")
        
        with col2:
            st.markdown("#### ▸ Panduan Input")
            st.info("""
            **Tips:**
            • Pastikan data akurat
            • Gunakan alat ukur standar
            • Ukur di pagi hari
            • Catat dengan teliti
            """)
        
        submitted = st.form_submit_button("⚡ Analisis Status Gizi", use_container_width=True)
    
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
            
            # Show processing animation
            with st.spinner('⚡ AI sedang menganalisis data...'):
                # Prepare input data
                input_data = {
                    'Sex': 1 if gender == "Male" else 0,
                    'Age': age,
                    'Height': height,
                    'Weight': weight
                }
                
                input_df = pd.DataFrame([input_data])
                
                for feature in feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                input_df = input_df[feature_names]
                input_scaled = scaler.transform(input_df)
                
                # Predict
                prediction_proba = model.predict(input_scaled, verbose=0)[0]
                prediction_class = np.argmax(prediction_proba)
                risk_level = target_encoder.inverse_transform([prediction_class])[0]
            
            # Display results with modern cards
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>▸ Hasil Analisis</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Status Card with integrated design
            status_colors_bg = {
                'Underweight': 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)',
                'Stunting': 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)',
                'Normal': 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%)',
                'Overweight': 'linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%)'
            }
            status_colors_border = {
                'Underweight': '#ef4444',
                'Stunting': '#ef4444',
                'Normal': '#10b981',
                'Overweight': '#fbbf24'
            }
            status_colors_text = {
                'Underweight': '#dc2626',
                'Stunting': '#dc2626',
                'Normal': '#059669',
                'Overweight': '#d97706'
            }
            status_icons = {
                'Underweight': '⚠',
                'Stunting': '⚠',
                'Normal': '✓',
                'Overweight': '▲'
            }
            
            st.markdown(f"""
            <div style='
                background: {status_colors_bg.get(risk_level, status_colors_bg["Normal"])};
                padding: 2.5rem;
                border-radius: 20px;
                border: 2px solid {status_colors_border.get(risk_level, "#10b981")};
                box-shadow: 0 10px 30px {status_colors_border.get(risk_level, "#10b981")}33;
                margin-bottom: 2rem;
            '>
                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                    <div style='
                        font-size: 3rem; 
                        color: {status_colors_text.get(risk_level, "#059669")};
                        line-height: 1;
                    '>
                        {status_icons.get(risk_level, '◆')}
                    </div>
                    <div>
                        <div style='
                            font-size: 0.9rem; 
                            font-weight: 600; 
                            text-transform: uppercase; 
                            letter-spacing: 0.5px;
                            color: {status_colors_text.get(risk_level, "#059669")};
                            opacity: 0.8;
                            margin-bottom: 0.3rem;
                        '>
                            Status Gizi Anak
                        </div>
                        <div style='
                            font-size: 2.5rem; 
                            font-weight: 800; 
                            color: {status_colors_text.get(risk_level, "#059669")};
                            line-height: 1;
                        '>
                            {risk_level.upper()}
                        </div>
                    </div>
                </div>
                <div style='
                    font-size: 1.1rem; 
                    color: #334155; 
                    font-weight: 600;
                    margin-top: 1rem;
                    padding-top: 1rem;
                    border-top: 1px solid {status_colors_border.get(risk_level, "#10b981")}40;
                '>
                    <span style='opacity: 0.7;'>Tingkat Kepercayaan:</span> 
                    <span style='color: {status_colors_text.get(risk_level, "#059669")};'>{prediction_proba[prediction_class]*100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution
            st.markdown("#### ▸ Distribusi Probabilitas Semua Kategori")
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
            st.markdown("#### ▸ Data Input yang Digunakan")
            input_summary = pd.DataFrame({
                'Parameter': ['Umur', 'Jenis Kelamin', 'Tinggi Badan', 'Berat Badan'],
                'Nilai': [f'{age} bulan', gender, f'{height} cm', f'{weight} kg']
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
            
            # Recommendations berdasarkan status gizi
            st.markdown("### ▸ Rekomendasi & Penjelasan")
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
Sistem: NutriPredict AI
            """
            
            st.download_button(
                label="⬇ Download Laporan",
                data=report,
                file_name=f"laporan_prediksi_status_gizi_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"✕ Terjadi kesalahan saat prediksi: {str(e)}")
            st.info("Pastikan model sudah dilatih dengan benar. Jalankan train_model.py terlebih dahulu.")

# ==================== PAGE: EVALUASI MODEL ====================
elif menu == "Evaluasi Model":
    st.markdown("<div class='main-header'>▸ Evaluasi Model AI</div>", unsafe_allow_html=True)
    
    if metrics is None:
        st.error("✕ Metrics belum tersedia. Silakan jalankan training script terlebih dahulu!")
        st.stop()
    
    # Overall Metrics
    st.markdown("### ▸ Performa Model")
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
    st.markdown("### ▸ Confusion Matrix")
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
    st.markdown("### ▸ Metrik Per Kelas")
    
    metrics_df = pd.DataFrame({
        'Kelas': metrics['target_names'],
        'Precision': [f"{p:.4f}" for p in metrics['precision']],
        'Recall': [f"{r:.4f}" for r in metrics['recall']],
        'F1-Score': [f"{f:.4f}" for f in metrics['f1_score']]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # False Positives & False Negatives Analysis
    st.markdown("### ▸ Analisis False Positives & False Negatives")
    
    for i, class_name in enumerate(metrics['target_names']):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        with st.expander(f"▸ {class_name.upper()}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("True Positives", tp)
            col2.metric("False Positives", fp)
            col3.metric("False Negatives", fn)
            
            if class_name in ['Stunting', 'Underweight'] and fn > 0:
                st.error(f"⚠ CRITICAL: {fn} anak dengan status '{class_name}' diprediksi salah! Mereka butuh perhatian khusus.")
    
    # Training History (if available)
    try:
        st.markdown("---")
        st.markdown("### ▸ Training History")
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
elif menu == "Dashboard Monitoring":
    st.markdown("<div class='main-header'>▸ Dashboard Monitoring</div>", unsafe_allow_html=True)
    
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
    
    st.markdown("### ▸ Ringkasan Statistik Stunting di Aceh per tahun 2022")
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
        st.markdown("#### ▸ Distribusi Tingkat Risiko")
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
        st.markdown("#### ▸ Risiko per Wilayah")
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
    st.markdown("### ▸ Data Anak / Daerah Berisiko Tinggi")
    if 'Jumlah' in demo_data.columns:
        high_risk = demo_data[demo_data['Risiko'] == 'Tinggi'].sort_values('Jumlah', ascending=False).head(20)
    else:
        high_risk = demo_data[demo_data['Risiko'] == 'Tinggi'].head(20)

    st.dataframe(high_risk.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Download
    csv = demo_data.to_csv(index=False)
    st.download_button(
        label="⬇ Download Data Lengkap (CSV)",
        data=csv,
        file_name="monitoring_stunting.csv",
        mime="text/csv"
    )

# ==================== PAGE: TENTANG ====================
elif menu == "Tentang":
    st.markdown("<div class='main-header'>ℹ Tentang NutriPredict</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### ▸ Tujuan Sistem
    
    **NutriPredict** dikembangkan untuk:
    1. Memprediksi status gizi anak berdasarkan data antropometri
    2. Memberikan rekomendasi intervensi gizi yang personal dan tepat sasaran
    3. Membantu tenaga kesehatan melakukan deteksi dini malnutrisi
    4. Mendukung program nasional penurunan stunting hingga target 14%
    
    ### ▸ Teknologi & Metodologi
    
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
    
    ### ▸ Dataset
    
    Dataset yang digunakan: **Child Growth and Nutrition Dataset** dari Kaggle
    - Berisi data antropometri anak (Age, Height, Weight)
    - Jenis Kelamin dan Status Gizi dari berbagai negara
    - Total data: 1000+ records
    - Pembagian data: 80% training, 20% testing
    
    ### ▸ Tim Pengembang
    
    Sistem ini dikembangkan oleh mahasiswa Informatika Universitas Syiah Kuala:
    
    • Reyan Andrea (2208107010014)
    
    • Shafa Disya Aulia (2308107010002)
    
    • Dea Zasqia Pasaribu Malau (2308107010004)
    
    • Tasya Zahrani (2308107010006)
    
    ### ▸ Referensi
    
    1. WHO. (2021). Child Growth Standards and Anthropometric Indices
    2. UNICEF. (2023). Global Nutrition Report
    3. Kementerian Kesehatan RI. (2023). Profil Kesehatan Indonesia
    4. Roesli et al. (2021). Artificial Neural Networks untuk Klasifikasi Status Gizi Anak
    
    ---
    
    © 2025 NutriPredict - Kelompok 1 - Project UAS Kecerdasan Artificial - Universitas Syiah Kuala
    """)
    
    st.success("✓ Sistem ini dikembangkan sebagai bagian dari Project Akhir Mata Kuliah Kecerdasan Artificial - Prediksi Status Gizi Anak dengan Artificial Neural Network")