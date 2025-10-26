import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown
import plotly.express as px
import pandas as pd

# --- Sabitler ve Konfigürasyon ---
MODEL_PATH = 'garbage_classifier_model.h5'
DRIVE_FILE_ID = '1uB24DQqKSCzTKGSjBsyjc7IuBOiCy4pw'
CLASS_NAMES_DRIVE_ID = '1tL43bFPuXYmd4iQ2A8HZZTTq9mno1z1F'
CLASS_NAMES_FILE = 'class_names.txt'
IMG_SIZE = (224, 224)

# Sınıflandırma sonuçları için detaylı bilgiler ve renkler
CATEGORY_INFO = {
    "cardboard": {
        "name": "Karton",
        "color": "#0077b6",
        "icon": "📦",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Karton kutuları düzleştirerek hacimden tasarruf edin. Yağlı veya ıslak kartonlar geri dönüştürülemez.",
        "co2_saving": 0.25
    },
    "glass": {
        "name": "Cam",
        "color": "#2a9d8f",
        "icon": "🍾",
        "bin": "Yeşil Kutu (Cam)",
        "recyclable": True,
        "tip": "Cam şişe ve kavanozları kapaksız olarak atın. Kırık camları dikkatli bir şekilde ayırın.",
        "co2_saving": 0.15
    },
    "metal": {
        "name": "Metal",
        "color": "#e9c46a",
        "icon": "🥫",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Konserve ve içecek kutularını temizleyip ezerek atın. Alüminyum folyo da geri dönüştürülebilir.",
        "co2_saving": 0.30
    },
    "paper": {
        "name": "Kağıt",
        "color": "#f4a261",
        "icon": "📰",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Gazeteler, dergiler ve ofis kağıtları geri dönüştürülebilir. Islak veya kirli kağıtları atmayın.",
        "co2_saving": 0.20
    },
    "plastic": {
        "name": "Plastik",
        "color": "#e76f51",
        "icon": "🥤",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Plastik şişelerin kapaklarını ayrı atın. Tüm plastik ambalajlar geri dönüştürülebilir olmayabilir.",
        "co2_saving": 0.10
    },
    "trash": {
        "name": "Diğer/Çöp",
        "color": "#264653",
        "icon": "🗑️",
        "bin": "Siyah Kutu (Geri Dönüştürülemez)",
        "recyclable": False,
        "tip": "Bu tür atıklar genellikle yakılır veya depolanır. Mümkün olduğunca bu kategorideki atıkları azaltmaya çalışın.",
        "co2_saving": 0.0
    }
}

# Model ve Sınıf İsimlerini Yükle
@st.cache_resource
def download_assets():
    success = True
    
    # Model Dosyasını İndir
    if not os.path.exists(MODEL_PATH):
        try:
            gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            success = False

    # Sınıf İsimleri Dosyasını İndir
    if not os.path.exists(CLASS_NAMES_FILE):
        try:
            gdown.download(id=CLASS_NAMES_DRIVE_ID, output=CLASS_NAMES_FILE, quiet=False, fuzzy=True)
        except Exception as e:
            success = False

    return success

@st.cache_resource
def load_assets():
    if not download_assets():
        return None, None
    
    try:
        model = load_model(MODEL_PATH)
        with open(CLASS_NAMES_FILE, 'r') as f:
            lines = f.readlines()
            class_names = []
            for line in lines:
                line = line.strip()
                if line:
                    # Teachable Machine format: "0 cardboard" veya sadece "cardboard"
                    parts = line.split(' ', 1)
                    if len(parts) > 1:
                        class_names.append(parts[1].lower())
                    else:
                        class_names.append(parts[0].lower())
        
        class_map = class_names
        return model, class_map
    except Exception as e:
        return None, None

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_array, class_map):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class_key = class_map[predicted_class_idx].lower()
    return predicted_class_key, confidence

# Modeli Yükle
model, class_map = load_assets()

# CSS Stilleri
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
* {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0e4eb 100%);
}
.main-container {
    display: grid;
    grid-template-columns: 1.2fr 1.2fr 1fr;
    gap: 20px;
    height: calc(100vh - 200px);
    overflow: hidden;
}
.column {
    overflow-y: auto;
    padding: 20px;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.modal-button {
    width: 100%;
    padding: 12px;
    margin: 8px 0;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}
.modal-button-about {
    background: linear-gradient(135deg, #2a9d8f 0%, #1e7a7a 100%);
    color: white;
}
.modal-button-about:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(42, 157, 143, 0.3);
}
.modal-button-project {
    background: linear-gradient(135deg, #e76f51 0%, #d45a3a 100%);
    color: white;
}
.modal-button-project:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(231, 111, 81, 0.3);
}
.result-card {
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
    color: white;
}
.result-card h3, .result-card h4 {
    color: white;
    font-weight: 600;
    margin: 10px 0;
}
.stFileUploader {
    border: 3px dashed #2a9d8f;
    border-radius: 15px;
    padding: 30px;
    background-color: rgba(255, 255, 255, 0.7);
}
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}
.modal-content {
    background: white;
    padding: 30px;
    border-radius: 20px;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}
.modal-header {
    font-size: 24px;
    font-weight: 700;
    color: #264653;
    margin-bottom: 20px;
    border-bottom: 3px solid #2a9d8f;
    padding-bottom: 15px;
}
.modal-close-btn {
    background: #e76f51;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    margin-top: 20px;
}
.stats-box {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0e4eb 100%);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #2a9d8f;
}
.stats-label {
    font-size: 12px;
    color: #666;
    font-weight: 600;
    text-transform: uppercase;
}
.stats-value {
    font-size: 20px;
    font-weight: 700;
    color: #264653;
    margin-top: 5px;
}
</style>
''', unsafe_allow_html=True)

# Session State Başlatma
if 'show_project_modal' not in st.session_state:
    st.session_state.show_project_modal = False
if 'show_about_modal' not in st.session_state:
    st.session_state.show_about_modal = False
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
    st.session_state.prediction_counts = {k: 0 for k in CATEGORY_INFO.keys()}
    st.session_state.total_co2_saved = 0.0
    st.session_state.last_prediction = "Henüz analiz yapılmadı"

# Başlık
st.title("🌱 Akıllı Geri Dönüşüm Asistanı")

# 3 Sütunlu Layout
col_left, col_middle, col_right = st.columns([1.2, 1.2, 1], gap="medium")

# --- SOL SÜTUN: Görüntü Yükleme ve İstatistikler ---
with col_left:
    st.subheader("📸 Fotoğraf Yükle")
    uploaded_file = st.file_uploader(
        "Fotoğraf Yükle (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        help="Lütfen net ve tek bir atık içeren bir fotoğraf yükleyin.",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Yüklenen Fotoğraf', use_column_width=True)
    
    st.markdown("---")
    st.subheader("📊 İstatistikler")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-label">Toplam Analiz</div>
            <div class="stats-value">{st.session_state.analysis_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-label">CO2 Tasarrufu</div>
            <div class="stats-value">{st.session_state.total_co2_saved:.2f} kg</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-box">
        <div class="stats-label">Son Tahmin</div>
        <div class="stats-value">{st.session_state.last_prediction}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Kategori Dağılımı Grafiği
    if st.session_state.analysis_count > 0:
        st.markdown("**Kategori Dağılımı**")
        plot_data = {
            "Kategori": [CATEGORY_INFO[k]['name'] for k, v in st.session_state.prediction_counts.items() if v > 0],
            "Sayı": [v for v in st.session_state.prediction_counts.values() if v > 0],
            "Renk": [CATEGORY_INFO[k]['color'] for k, v in st.session_state.prediction_counts.items() if v > 0]
        }
        if plot_data["Kategori"]:
            df = pd.DataFrame(plot_data)
            fig = px.pie(
                df, 
                values='Sayı', 
                names='Kategori', 
                color='Kategori',
                color_discrete_map={k: v for k, v in zip(df['Kategori'], df['Renk'])},
                hole=.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig, use_container_width=True)

# --- ORTA SÜTUN: Analiz Sonuçları ---
with col_middle:
    st.subheader("📊 Analiz Sonuçları")
    
    if uploaded_file is not None and model is not None:
        try:
            image_pil = Image.open(uploaded_file).convert('RGB')
            img_array = preprocess_image(image_pil)
            predicted_class_key, confidence = predict_image(model, img_array, class_map)
            info = CATEGORY_INFO.get(predicted_class_key, CATEGORY_INFO['trash'])
            
            # Session State'i güncelle
            st.session_state.analysis_count += 1
            st.session_state.prediction_counts[predicted_class_key] += 1
            st.session_state.total_co2_saved += info.get('co2_saving', 0.0)
            st.session_state.last_prediction = info['name']

            # Sonuç Kartı
            st.markdown(f'''
                <div class="result-card" style="background-color: {info['color']};">
                    <h3>🎯 TESPİT: {info['icon']} {info['name'].upper()}</h3>
                    <p>📊 Güven Skoru: <b>{confidence*100:.2f}%</b></p>
                    <hr style="border-color: white;">
                    <h4>📦 GERİ DÖNÜŞÜM BİLGİSİ</h4>
                    <p><b>Kutu:</b> {info['bin']}</p>
                    <p><b>Durum:</b> {'✅ GERİ DÖNÜŞTÜRÜLEBİLİR' if info['recyclable'] else '❌ GERİ DÖNÜŞTÜRÜLEMEZ'}</p>
                    <p><b>💡 İpucu:</b> {info['tip']}</p>
                    <hr style="border-color: white; opacity: 0.5;">
                    <p>🌍 Tahmini Çevresel Katkı: <b>{info['co2_saving']:.2f} kg CO2 tasarrufu</b></p>
                </div>
            ''', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")
    else:
        st.info("📸 Lütfen sol taraftan bir fotoğraf yükleyin.")

# --- SAĞ SÜTUN: Menü Butonları ---
with col_right:
    st.subheader("⚙️ Menü")
    
    # Hakkımızda Butonu
    if st.button("👤 Hakkımızda", key="about_btn", use_container_width=True):
        if st.session_state.show_about_modal:
            st.session_state.show_about_modal = False
        else:
            st.session_state.show_about_modal = True
            st.session_state.show_project_modal = False
        st.rerun()
    
    # Proje Hakkında Butonu
    if st.button("📊 Proje Hakkında", key="project_btn", use_container_width=True):
        if st.session_state.show_project_modal:
            st.session_state.show_project_modal = False
        else:
            st.session_state.show_project_modal = True
            st.session_state.show_about_modal = False
        st.rerun()

# --- MODAL İÇERİKLERİ ---

# Hakkımızda Modal
if st.session_state.show_about_modal:
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); display: flex; justify-content: center; align-items: center; z-index: 1000;">
        <div style="background: white; padding: 40px; border-radius: 20px; max-width: 600px; width: 90%; max-height: 85vh; overflow-y: auto; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        st.image("https://media.licdn.com/dms/image/v2/D4D03AQGVv9aJFngyfA/profile-displayphoto-crop_800_800/B4DZiTDBBMHsAI-/0/1754813700473?e=1762992000&v=beta&t=dHNPVRx59o4FH5GVoVZWgReb7R364ncx4lwhc32A6pM", width=150)
    
    st.markdown("### 👤 Hakkımızda")
    
    st.markdown("""
    **Alkım Can KALYONCU**
    
    🎓 **Ünvan:** Matematik Öğretmeni
    
    **Hakkında:**
    - 📚 Milli Eğitim Bakanlığı (MEB) Matematik Öğretmeni
    - 👨‍👦 Eymen Efe'nin babası
    - 🤖 KuGeN Takım Danışmanı
    - 🏆 2022 Teknofest Şampiyonu
    - 💪 "Düşse de kalkan - Vazgeçmeyen Adam"
    
    **İletişim:**
    - 📧 [alkimkalyoncu@gmail.com](mailto:alkimkalyoncu@gmail.com)
    - 💼 [LinkedIn](https://linkedin.com/in/alkım-can-kalyoncu-8433121a2)
    - 💻 [GitHub](https://github.com/alkimcan)
    - 📸 [Instagram](https://www.instagram.com/alkimkalyoncu/)
    
    ---
    
    **🌟 Biyografi**
    
    1980'lerin sonunda doğan, 2000'lerin başında "bu çocuk matematiği çözdü" etiketiyle büyüyen Alkım Can Kalyoncu, kendini sadece rakamların değil, fikirlerin ve hayallerin de ustası olarak kanıtladı. 📚✏️
    
    Matematik öğretmeni olarak başladığı yolculuğunu, gençlerin sadece dört işlem değil; robotik, yapay zekâ, 3D tasarım ve hayallerle tanışabileceği atölyelere dönüştürdü. 🚀 KuGeN Maker Teknoloji ve Akademik Danışmanlık Merkezi ile Boyabat'ın kalbine küçük bir "gelecek fabrikası" kurdu. Burada öğrenciler sadece sınava değil, geleceğe hazırlanıyor.
    
    Ama Alkım'ı sadece ders anlatan biri sanmayın. 🎭 O, bir yandan Sinop'un 1214'teki fethini kısa filmlere konu eden; Refik Anadol esintileriyle "makine rüyaları" tasarlayan; öte yandan Instagram'da kitaplarıyla #okudumbitti paylaşımları yaparak "modern zaman hikâye anlatıcısı"na dönüşen bir karakter.
    
    Kendi tabiriyle:
    
    **"Ben sadece bir öğretmen değilim; aynı zamanda hayallerini projeye, projelerini esere, eserlerini de gençlerin hayatına işleyen bir yol arkadaşıyım."** 🌌
    
    Bugünlerde hedefi; hem matematikte hem teknolojide hem de hayatta gençlere "yapabilirsin" dedirtmek. Ve elbette, biraz da kendi hayatını baştan yazmak. ✨
    """)
    
    if st.button("Kapat", key="close_about"):
        st.session_state.show_about_modal = False
        st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Proje Hakkında Modal
if st.session_state.show_project_modal:
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); display: flex; justify-content: center; align-items: center; z-index: 1000;">
        <div style="background: white; padding: 40px; border-radius: 20px; max-width: 600px; width: 90%; max-height: 85vh; overflow-y: auto; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);">
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Proje Hakkında")
    
    st.markdown("""
    **Akıllı Geri Dönüşüm Asistanı**, yapay zeka ve görüntü sınıflandırma teknolojisini kullanarak 
    atıkları otomatik olarak kategorize eden bir uygulamadır.
    
    **Proje Özellikleri:**
    - 🤖 Teachable Machine ile eğitilmiş model (%85-90 doğruluk)
    - 📊 Gerçek zamanlı istatistik ve analiz
    - 🌍 CO2 tasarrufu hesaplaması
    - 💡 Geri dönüşüm ipuçları ve bilgileri
    
    **Teknoloji Stack:**
    - Streamlit (Web Arayüzü)
    - TensorFlow/Keras (Model)
    - Google Drive (Veri Depolama)
    - Plotly (Grafikler)
    
    **Veri Seti:**
    - Kaggle 'Garbage Classification' (5,054 görsel)
    - 6 Atık Kategorisi
    
    **Proje Hedefleri:**
    - ♻️ Geri dönüşüm bilincini artırmak
    - 🌍 Çevresel etki azaltmak
    - 🤖 Yapay zeka teknolojisini yaygınlaştırmak
    """)
    
    if st.button("Kapat", key="close_project"):
        st.session_state.show_project_modal = False
        st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("✅ **Model Bilgisi:** Bu uygulama, Teachable Machine ile eğitilmiş ve **%85-90 doğruluk** elde eden bir model kullanmaktadır.")

