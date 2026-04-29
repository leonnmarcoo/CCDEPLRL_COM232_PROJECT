import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Adjust these labels based on your dataset's exact alphabetical folder names
# You can check this by printing: train_data.class_indices in your notebook
CLASS_NAMES = ['Improperly Worn', 'Unmasked', 'Masked'] 

@st.cache_resource
def load_trained_model():
    # Load the model saved from the notebook
    return tf.keras.models.load_model('mask_model.h5')

st.title("Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask properly.")

# Try to load the model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Could not load the model. Please make sure you have saved it as 'mask_model.h5'.\nError: {e}")
    st.stop()

# Input options
option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

input_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')
elif option == "Use Camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert('RGB')

if input_image is not None:
    # Display the image
    st.image(input_image, caption='Image for Detection', use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess image to match training exactly (128x128, scaled by 1/255)
    img = input_image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    
    st.success(f"Prediction: **{predicted_class_name}** ({confidence*100:.2f}%)")

