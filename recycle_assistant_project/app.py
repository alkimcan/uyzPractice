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
        "color": "#0077b6", # Mavi
        "icon": "📦",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Karton kutuları düzleştirerek hacimden tasarruf edin. Yağlı veya ıslak kartonlar geri dönüştürülemez.",
        "co2_saving": 0.25
    },
    "glass": {
        "name": "Cam",
        "color": "#2a9d8f", # Yeşilimsi Mavi
        "icon": "🍾",
        "bin": "Yeşil Kutu (Cam)",
        "recyclable": True,
        "tip": "Cam şişe ve kavanozları kapaksız olarak atın. Kırık camları dikkatli bir şekilde ayırın.",
        "co2_saving": 0.15
    },
    "metal": {
        "name": "Metal",
        "color": "#e9c46a", # Sarımsı
        "icon": "🥫",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Konserve ve içecek kutularını temizleyip ezerek atın. Alüminyum folyo da geri dönüştürülebilir.",
        "co2_saving": 0.30
    },
    "paper": {
        "name": "Kağıt",
        "color": "#f4a261", # Turuncu
        "icon": "📰",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Gazeteler, dergiler ve ofis kağıtları geri dönüştürülebilir. Islak veya kirli kağıtları atmayın.",
        "co2_saving": 0.20
    },
    "plastic": {
        "name": "Plastik",
        "color": "#e76f51", # Kırmızımsı
        "icon": "🥤",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Plastik şişelerin kapaklarını ayrı atın. Tüm plastik ambalajlar geri dönüştürülebilir olmayabilir.",
        "co2_saving": 0.10
    },
    "trash": {
        "name": "Diğer/Çöp",
        "color": "#264653", # Koyu Mavi
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
        st.info("Model dosyası bulunamadı. Google Drive'dan indiriliyor...")
        try:
            gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("Model başarıyla indirildi!")
        except Exception as e:
            st.error(f"Model indirilirken hata oluştu: {e}")
            success = False

    # Sınıf İsimleri Dosyasını İndir
    if not os.path.exists(CLASS_NAMES_FILE):
        st.info("Sınıf isimleri dosyası bulunamadı. Google Drive'dan indiriliyor...")
        try:
            gdown.download(id=CLASS_NAMES_DRIVE_ID, output=CLASS_NAMES_FILE, quiet=False)
            st.success("Sınıf isimleri dosyası başarıyla indirildi!")
        except Exception as e:
            st.error(f"Sınıf isimleri dosyası indirilirken hata oluştu: {e}")
            success = False

    return success

@st.cache_resource
def load_assets():
    if not download_assets():
        return None, None
    
    try:
        model = load_model(MODEL_PATH)
        with open(CLASS_NAMES_FILE, 'r') as f:
            # Teachable Machine labels.txt dosyasında "0 cardboard" gibi format olduğu için
            # sadece sınıf ismini alacak şekilde düzenliyoruz.
            class_names = [line.strip().split(' ', 1)[1] for line in f.readlines() if line.strip()]
        
        # Teachable Machine'den gelen class_names listesi zaten doğru sıradadır.
        class_map = class_names
        
        return model, class_map
    except Exception as e:
        st.error(f"Varlıklar yüklenirken bir hata oluştu: {e}")
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
html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0e4eb 100%);
}
.st-emotion-cache-1cypcdb { 
    color: #264653;
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.result-card {
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
    color: white;
    transition: transform 0.3s ease-in-out;
}
.result-card:hover {
    transform: translateY(-5px);
}
.result-card h3, .result-card h4 {
    color: white;
    font-weight: 600;
}
.stFileUploader {
    border: 3px dashed #2a9d8f;
    border-radius: 15px;
    padding: 30px;
    background-color: rgba(255, 255, 255, 0.7);
}
.stButton>button {
    background-color: #2a9d8f;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}
.sidebar-button {
    margin: 10px 0;
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
st.markdown("Yüklediğiniz atık görselini analiz ederek hangi kategoriye ait olduğunu ve nasıl geri dönüştürüleceğini öğrenin.")

# 3 Sütunlu Layout
col_left, col_middle, col_right = st.columns([1.5, 1.5, 1.2])

# --- SOL SÜTUN: Görüntü Yükleme ---
with col_left:
    st.subheader("📸 Fotoğraf Yükle")
    uploaded_file = st.file_uploader(
        "Fotoğraf Yükle (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        help="Lütfen net ve tek bir atık içeren bir fotoğraf yükleyin."
    )
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='Yüklenen Fotoğraf', use_column_width=True)

# --- ORTA SÜTUN: Analiz Sonuçları ---
with col_middle:
    st.subheader("📊 Analiz Sonuçları")
    
    if uploaded_file is not None and model is not None:
        try:
            image_pil = Image.open(uploaded_file).convert('RGB')
            st.info("Analiz ediliyor...")
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
        st.info("Lütfen sol taraftan bir fotoğraf yükleyin.")

# --- SAĞ SÜTUN: Butonlar ve İstatistikler ---
with col_right:
    st.subheader("⚙️ Menü")
    
    # Hakkımızda Butonu
    if st.button("👤 Hakkımızda", use_container_width=True, key="about_btn"):
        if st.session_state.show_about_modal:
            st.session_state.show_about_modal = False
        else:
            st.session_state.show_about_modal = True
            st.session_state.show_project_modal = False
    
    st.markdown("")  # Boşluk
    
    # Proje Hakkında Butonu
    if st.button("📊 Proje Hakkında", use_container_width=True, key="project_btn"):
        if st.session_state.show_project_modal:
            st.session_state.show_project_modal = False
        else:
            st.session_state.show_project_modal = True
            st.session_state.show_about_modal = False

# --- MODAL İÇERİKLERİ ---

# Hakkımızda Modal
if st.session_state.show_about_modal:
    st.markdown("---")
    st.markdown("### 👤 Hakkımızda")
    
    # Profil Fotoğrafı
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        st.image("https://media.licdn.com/dms/image/v2/D4D03AQGVv9aJFngyfA/profile-displayphoto-crop_800_800/B4DZiTDBBMHsAI-/0/1754813700473?e=1762992000&v=beta&t=dHNPVRx59o4FH5GVoVZWgReb7R364ncx4lwhc32A6pM", width=150)
    
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

# Proje Hakkında Modal
if st.session_state.show_project_modal:
    st.markdown("---")
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
    
    **Proje Sunumu:** Detaylı bilgi ve analiz için sidebar'daki butonları kullanın.
    """)

# --- Sidebar ve İstatistikler ---
st.sidebar.header("📊 Analiz İstatistikleri")

# Toplam Analiz ve CO2 Metrikleri
st.sidebar.metric(label="Toplam Analiz Sayısı", value=st.session_state.analysis_count)
st.sidebar.metric(label="Tahmini CO2 Tasarrufu", value=f"{st.session_state.total_co2_saved:.2f} kg", delta="Geri Dönüşüm Katkınız")
st.sidebar.info(f"Son Tahmin: {st.session_state.last_prediction}")

st.sidebar.markdown("---")

# Plotly Pasta Grafiği
if st.session_state.analysis_count > 0:
    st.sidebar.subheader("Kategori Dağılımı")
    plot_data = {
        "Kategori": [CATEGORY_INFO[k]['name'] for k, v in st.session_state.prediction_counts.items() if v > 0],
        "Sayı": [v for v in st.session_state.prediction_counts.values() if v > 0],
        "Renk": [CATEGORY_INFO[k]['color'] for k, v in st.session_state.prediction_counts.items() if v > 0]
    }
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
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
    st.sidebar.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Bu uygulama, Kaggle 'Garbage Classification' veri seti kullanılarak eğitilmiş bir model ile güçlendirilmiştir.")

# Modelin doğruluk oranını kullanıcıya bildiren not
if model is not None:
    st.success("✅ **Model Bilgisi:** Bu uygulama, Teachable Machine ile eğitilmiş ve **%85-90 doğruluk** elde eden bir model kullanmaktadır. Üretim ortamı için optimize edilmiştir.")

# Footer
st.markdown("---")
st.markdown("Geliştirme aşaması: **İyileştirme ve Test**")

