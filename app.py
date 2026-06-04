import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

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

    .live-status {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .live-status-masked {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .live-status-unmasked {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .live-status-improper {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 2px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Display names mapped from dataset folder names
# YOLOv8 sorts class folders alphabetically, so the order is:
#   0: Improperly Wearing Facemask
#   1: Not Wearing Facemask
#   2: Wearing Facemask
DISPLAY_NAMES = {
    'Improperly Wearing Facemask': 'Improperly Worn',
    'Not Wearing Facemask': 'Unmasked',
    'Wearing Facemask': 'Masked',
}

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
    return YOLO('best.pt')

with st.sidebar:
    st.markdown("### 📋 About")
    st.markdown(
        "This app uses a **YOLOv8 Classification Model** "
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
        "1. Upload a photo, take one with your camera, **or use live detection**\n"
        "2. The image is resized to 128×128 pixels\n"
        "3. The YOLOv8 model analyzes the image\n"
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
        f"⚠️ Could not load the model. Make sure `best.pt` "
        f"is in the project directory.\n\n**Error:** {e}"
    )
    st.stop()

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = 'upload'

col1, col2, col3 = st.columns(3)

with col1:
    upload_tab = st.button("📁 Upload Image", use_container_width=True, type="primary" if st.session_state.input_mode == 'upload' else "secondary")
with col2:
    camera_tab = st.button("📷 Use Camera", use_container_width=True, type="primary" if st.session_state.input_mode == 'camera' else "secondary")
with col3:
    live_tab = st.button("🎥 Live Detection", use_container_width=True, type="primary" if st.session_state.input_mode == 'live' else "secondary")

if upload_tab:
    st.session_state.input_mode = 'upload'
    st.rerun()
if camera_tab:
    st.session_state.input_mode = 'camera'
    st.rerun()
if live_tab:
    st.session_state.input_mode = 'live'
    st.rerun()

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
elif st.session_state.input_mode == 'camera':
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert('RGB')
elif st.session_state.input_mode == 'live':
    st.markdown("#### 🎥 Real-Time Face Mask Detection")
    st.caption("Your webcam feed is classified frame-by-frame. The overlay shows the current prediction.")

    # Color map for overlay: BGR format for OpenCV
    OVERLAY_COLORS = {
        'Masked': (40, 167, 69),         # green
        'Unmasked': (220, 53, 69),       # red
        'Improperly Worn': (255, 193, 7) # yellow
    }

    import cv2

    class FaceMaskProcessor(VideoProcessorBase):
        def __init__(self):
            self._model = load_trained_model()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Convert BGR -> RGB PIL image for YOLO
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            results = self._model.predict(pil_img, imgsz=128, verbose=False)
            probs = results[0].probs

            raw_name = self._model.names[probs.top1]
            display_name = DISPLAY_NAMES.get(raw_name, raw_name)
            confidence = float(probs.top1conf)

            # Choose overlay color
            color = OVERLAY_COLORS.get(display_name, (255, 255, 255))

            # Draw semi-transparent banner at the top
            h, w = img.shape[:2]
            overlay = img.copy()
            banner_h = 70
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

            # Draw label text
            icon_text = {'Masked': 'MASKED', 'Unmasked': 'NO MASK', 'Improperly Worn': 'IMPROPER'}  
            label = f"{icon_text.get(display_name, display_name)}  {confidence*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (banner_h + text_size[1]) // 2

            # Shadow for readability
            cv2.putText(img, label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Thin colored border around the entire frame
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 4)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="facemask-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceMaskProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if input_image is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    img_col, result_col = st.columns([1, 1], gap="large")
    
    with img_col:
        st.markdown("#### 🖼️ Input Image")
        st.image(input_image, use_container_width=True)
    
    with result_col:
        st.markdown("#### 🔍 Analysis Result")
        
        with st.spinner("Analyzing..."):
            # YOLOv8 classification prediction
            results = model.predict(input_image, imgsz=128, verbose=False)
            probs = results[0].probs
            
            # Get the raw class name from model and map to display name
            raw_class_name = model.names[probs.top1]
            predicted_class_name = DISPLAY_NAMES.get(raw_class_name, raw_class_name)
            confidence = float(probs.top1conf)
            
            # Build ordered class info for the breakdown
            class_probs = []
            for idx, raw_name in model.names.items():
                display_name = DISPLAY_NAMES.get(raw_name, raw_name)
                prob = float(probs.data[idx])
                class_probs.append((display_name, prob))
        
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
            for class_name, prob in class_probs:
                st.progress(float(prob), text=f"{CLASS_ICONS.get(class_name, '')} {class_name}: {prob*100:.1f}%")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #aaa; font-size: 0.85rem;'>"
    "Built with Streamlit • YOLOv8 Face Mask Detection Project"
    "</p>",
    unsafe_allow_html=True
)
