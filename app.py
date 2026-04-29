import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #667eea;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }
    
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-masked {
        background: #d4edda;
        border: 2px solid #28a745;
    }
    .result-unmasked {
        background: #f8d7da;
        border: 2px solid #dc3545;
    }
    .result-improper {
        background: #fff3cd;
        border: 2px solid #ffc107;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #222;
        margin-bottom: 0.3rem;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #555;
    }
    
    .custom-divider {
        height: 2px;
        background: #ddd;
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['Improperly Worn', 'Unmasked', 'Masked']

CLASS_ICONS = {
    'Masked': '✅',
    'Unmasked': '❌',
    'Improperly Worn': '⚠️',
}

CLASS_COLORS = {
    'Masked': 'result-masked',
    'Unmasked': 'result-unmasked',
    'Improperly Worn': 'result-improper',
}

CLASS_MESSAGES = {
    'Masked': 'The person is wearing a face mask properly.',
    'Unmasked': 'The person is NOT wearing a face mask.',
    'Improperly Worn': 'The face mask is not worn correctly (e.g., below the nose or chin).',
}

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('mask_model.h5')

with st.sidebar:
    st.markdown("### 📋 About")
    st.markdown(
        "This app uses a **Convolutional Neural Network (CNN)** "
        "to detect whether a person is wearing a face mask properly."
    )
    
    st.divider()
    
    st.markdown("### 🏷️ Categories")
    st.markdown(
        "- ✅ **Masked** — Properly worn\n"
        "- ⚠️ **Improperly Worn** — Below nose/chin\n"
        "- ❌ **Unmasked** — No mask detected"
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ How It Works")
    st.markdown(
        "1. Upload a photo or take one with your camera\n"
        "2. The image is resized to 128×128 pixels\n"
        "3. The CNN analyzes the image\n"
        "4. A prediction with confidence score is displayed"
    )

st.markdown("""
<div class="main-header">
    <h1>😷 Face Mask Detector</h1>
    <p>Upload or capture an image to classify mask usage</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

try:
    model = load_trained_model()
except Exception as e:
    st.error(
        f"⚠️ Could not load the model. Make sure `mask_model.h5` "
        f"is in the project directory.\n\n**Error:** {e}"
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    upload_tab = st.button("📁 Upload Image", use_container_width=True, type="primary")
with col2:
    camera_tab = st.button("📷 Use Camera", use_container_width=True)

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = 'upload'

if upload_tab:
    st.session_state.input_mode = 'upload'
if camera_tab:
    st.session_state.input_mode = 'camera'

st.markdown("")

input_image = None

if st.session_state.input_mode == 'upload':
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Supported formats: JPG, JPEG, PNG"
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert('RGB')

if input_image is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    img_col, result_col = st.columns([1, 1], gap="large")
    
    with img_col:
        st.markdown("#### 🖼️ Input Image")
        st.image(input_image, use_container_width=True)
    
    with result_col:
        st.markdown("#### 🔍 Analysis Result")
        
        with st.spinner("Analyzing..."):
            img = input_image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
        icon = CLASS_ICONS.get(predicted_class_name, '🔍')
        color_class = CLASS_COLORS.get(predicted_class_name, 'result-masked')
        message = CLASS_MESSAGES.get(predicted_class_name, '')
        
        st.markdown(f"""
        <div class="result-card {color_class}">
            <div class="result-label">{icon} {predicted_class_name}</div>
            <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        st.info(f"💡 {message}")
        
        with st.expander("📊 Confidence Breakdown"):
            for i, class_name in enumerate(CLASS_NAMES):
                prob = predictions[0][i]
                st.progress(float(prob), text=f"{CLASS_ICONS.get(class_name, '')} {class_name}: {prob*100:.1f}%")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #aaa; font-size: 0.85rem;'>"
    "Built with Streamlit • CNN Face Mask Detection Project"
    "</p>",
    unsafe_allow_html=True
)
