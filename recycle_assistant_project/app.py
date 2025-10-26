import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- Sabitler ve Konfigürasyon ---
MODEL_PATH = 'garbage_classifier_model.h5'
CLASS_NAMES_FILE = 'class_names.txt'
IMG_SIZE = (224, 224)

# Sınıflandırma sonuçları için detaylı bilgiler ve renkler
# Modelin sınıf isimleri: ['Garbage classification', 'cardboard', 'garbage classification', 'glass', 'metal', 'paper', 'plastic']
# Bu isimleri kullanıcı dostu hale getiriyoruz.
# Not: "Garbage classification" ve "garbage classification" aynı sınıfı temsil ediyor, bu yüzden "Diğer/Çöp" olarak birleştiriyoruz.
CATEGORY_INFO = {
    "cardboard": {
        "name": "Karton",
        "color": "#0077b6", # Mavi (Planınızdaki renk)
        "icon": "📦",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Karton kutuları düzleştirerek hacimden tasarruf edin. Yağlı veya ıslak kartonlar geri dönüştürülemez.",
        "co2_saving": 0.25 # Örnek değer
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
    "cardboard": {
        "name": "Karton",
        "color": "orange",
        "icon": "📦",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Karton kutuları düzleştirerek hacimden tasarruf edin. Yağlı veya ıslak kartonlar geri dönüştürülemez."
    },
    "glass": {
        "name": "Cam",
        "color": "blue",
        "icon": "🍾",
        "bin": "Yeşil Kutu (Cam)",
        "recyclable": True,
        "tip": "Cam şişe ve kavanozları kapaksız olarak atın. Kırık camları dikkatli bir şekilde ayırın."
    },
    "metal": {
        "name": "Metal",
        "color": "gray",
        "icon": "🥫",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Konserve ve içecek kutularını temizleyip ezerek atın. Alüminyum folyo da geri dönüştürülebilir."
    },
    "paper": {
        "name": "Kağıt",
        "color": "green",
        "icon": "📰",
        "bin": "Mavi Kutu (Kağıt/Karton)",
        "recyclable": True,
        "tip": "Gazeteler, dergiler ve ofis kağıtları geri dönüştürülebilir. Islak veya kirli kağıtları atmayın."
    },
    "plastic": {
        "name": "Plastik",
        "color": "red",
        "icon": "🥤",
        "bin": "Sarı Kutu (Metal/Plastik)",
        "recyclable": True,
        "tip": "Plastik şişelerin kapaklarını ayrı atın. Tüm plastik ambalajlar geri dönüştürülebilir olmayabilir."
    },
    "trash": {
        "name": "Diğer/Çöp",
        "color": "black",
        "icon": "🗑️",
        "bin": "Siyah Kutu (Geri Dönüştürülemez)",
        "recyclable": False,
        "tip": "Bu tür atıklar genellikle yakılır veya depolanır. Mümkün olduğunca bu kategorideki atıkları azaltmaya çalışın."
    }
}

# Model ve Sınıf İsimlerini Yükle
@st.cache_resource
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        with open(CLASS_NAMES_FILE, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        # Sınıfları temizle (Çift olanları tek bir "trash" sınıfına eşitle)
        cleaned_class_names = []
        for name in class_names:
            if name.lower() == 'garbage classification':
                if 'trash' not in cleaned_class_names:
                    cleaned_class_names.append('trash')
            else:
                cleaned_class_names.append(name.lower())
        
        # Sınıfları alfabetik olarak sırala (Modelin eğitim sırasını korumak için)
        # Modelin eğitim sırası: ['Garbage classification', 'cardboard', 'garbage classification', 'glass', 'metal', 'paper', 'plastic']
        # Bu sırayı koruyarak bir eşleme yapmalıyız.
        
        # Basitçe, modelin tahmin ettiği indeksi, CATEGORY_INFO'daki anahtarlara eşleyelim.
        # Bu, modelin eğitim sırasına bağlıdır.
        # 0: Garbage classification (trash)
        # 1: cardboard
        # 2: garbage classification (trash)
        # 3: glass
        # 4: metal
        # 5: paper
        # 6: plastic
        
        # Eşleme listesi:
        class_map = ['trash', 'cardboard', 'trash', 'glass', 'metal', 'paper', 'plastic']
        
        return model, class_map
    except Exception as e:
        st.error(f"Varlıklar yüklenirken bir hata oluştu: {e}")
        return None, None

model, class_map = load_assets()

# --- Görüntü İşleme Fonksiyonu ---
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2 kullandığımız için 0-255 aralığında bırakıp, model içinde ön işleme yapılmasını bekliyoruz.
    # Ancak eğitimde 1/255 ile rescale ettiğimiz için burada da yapmalıyız.
    img_array /= 255.0
    return img_array

# --- Tahmin Fonksiyonu ---
def predict_image(model, img_array, class_map):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # En yüksek skorlu tahmini bul
    predicted_index = np.argmax(score)
    confidence = np.max(score)
    
    # Sınıf adını eşleme listesinden al
    predicted_class_key = class_map[predicted_index]
    
    return predicted_class_key, confidence

# --- Streamlit Arayüzü ---

st.set_page_config(
    page_title="🌱 Akıllı Geri Dönüşüm Asistanı",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Planınızdaki modern tasarım isteğine uygun: Gradient background, modern font)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0e4eb 100%); /* Hafif Gradient background */
}

.st-emotion-cache-1cypcdb { /* Ana başlık */
    color: #264653; /* Koyu renk */
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

</style>
""", unsafe_allow_html=True)

# Başlık
st.title("🌱 Akıllı Geri Dönüşüm Asistanı")
st.markdown("Yüklediğiniz atık görselini analiz ederek hangi kategoriye ait olduğunu ve nasıl geri dönüştürüleceğini öğrenin.")

# --- Ana İçerik ve Yükleyici ---
uploaded_file = st.file_uploader(
    "Fotoğraf Yükle (PNG, JPG)",
    type=["png", "jpg", "jpeg"],
    help="Lütfen net ve tek bir atık içeren bir fotoğraf yükleyin."
)

if uploaded_file is not None and model is not None:
    # Görüntüyü aç ve ön işle
    try:
        image_pil = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image_pil, caption='Yüklenen Fotoğraf', use_column_width=True)
            
        with col2:
            st.info("Analiz ediliyor...")
            
            # Tahmin yap
            img_array = preprocess_image(image_pil)
            predicted_class_key, confidence = predict_image(model, img_array, class_map)
            
            # Sonuç Bilgilerini Al
            info = CATEGORY_INFO.get(predicted_class_key, CATEGORY_INFO['trash'])
            
            # Sonuç Kartı (Planınızdaki renkli kartlar)
            st.markdown(f"""
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
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analiz sırasında bir hata oluştu: {e}")

# --- Sidebar ve İstatistikler ---
# Basit bir istatistik paneli (Planınızdaki 7. madde)
st.sidebar.header("📊 Analiz İstatistikleri")
# Session State Başlatma
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
    st.session_state.prediction_counts = {k: 0 for k in CATEGORY_INFO.keys()}
    st.session_state.total_co2_saved = 0.0
    st.session_state.last_prediction = "Henüz analiz yapılmadı"

# Tahmin yapıldıktan sonra session state'i güncelle
if uploaded_file is not None and model is not None:
    # Her başarılı analizde sayacı artır
    if 'predicted_class_key' in locals():
        st.session_state.analysis_count += 1
        st.session_state.prediction_counts[predicted_class_key] += 1
        st.session_state.total_co2_saved += info.get('co2_saving', 0.0)
        st.session_state.last_prediction = info['name']

# --- Sidebar ve İstatistikler (Plotly ile) ---
st.sidebar.header("📊 Analiz İstatistikleri")

# Toplam Analiz ve CO2 Metrikleri
st.sidebar.metric(label="Toplam Analiz Sayısı", value=st.session_state.analysis_count)
st.sidebar.metric(label="Tahmini CO2 Tasarrufu", value=f"{st.session_state.total_co2_saved:.2f} kg", delta="Geri Dönüşüm Katkınız")
st.sidebar.info(f"Son Tahmin: {st.session_state.last_prediction}")

st.sidebar.markdown("---")

# Plotly Pasta Grafiği (Kategori Dağılımı)
if st.session_state.analysis_count > 0:
    st.sidebar.subheader("Kategori Dağılımı")
    
    # Sadece tahmin yapılan kategorileri al
    plot_data = {
        "Kategori": [CATEGORY_INFO[k]["name"] for k, v in st.session_state.prediction_counts.items() if v > 0],
        "Sayı": [v for v in st.session_state.prediction_counts.values() if v > 0],
        "Renk": [CATEGORY_INFO[k]["color"] for k, v in st.session_state.prediction_counts.items() if v > 0]
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
st.sidebar.caption("Bu uygulama, Kaggle 'Garbage Classification' veri seti kullanılarak eğitilmiş bir MobileNetV2 modeli ile güçlendirilmiştir.")

# Dark Mode/Light Mode Toggle (Basit bir tema değiştirme)
st.sidebar.subheader("Tema")
if st.sidebar.button("🌙 Dark Mode'a Geç"):
    st.session_state.theme = 'dark'
    st.experimental_rerun()
elif st.sidebar.button("☀️ Light Mode'a Geç"):
    st.session_state.theme = 'light'
    st.experimental_rerun()

if 'theme' in st.session_state and st.session_state.theme == 'dark':
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
        }
        .st-emotion-cache-1cypcdb {
            color: #2a9d8f;
        }
        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True
    )

# Modelin düşük doğruluk oranını kullanıcıya bildiren not
if model is not None:
    st.warning("⚠️ **Önemli Not:** Model, kısıtlı eğitim süresi nedeniyle yaklaşık **%52 eğitim ve %30 doğrulama doğruluğuna** sahiptir. Tahminlerinizde hatalar olabilir. Daha yüksek doğruluk için ek eğitim gereklidir.")

# Footer
st.markdown("---")
st.markdown("Geliştirme aşaması: **Streamlit Uygulaması Geliştirme**")
