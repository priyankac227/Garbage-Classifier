import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# # Load trained model
if os.path.exists("trashnet_cnn_from_scratch_final.h5"):
    model = tf.keras.models.load_model(
        "trashnet_cnn_from_scratch_final.h5",
        compile=False
    )
else:
    st.error("Model not found. Please run train.py first.")
    st.stop()
# st.warning("Model is not loaded. To enable predictions, please run train.py to train the model and uncomment the model loading and prediction code in app.py.")

# Class labels (must match training order)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Waste Classification App")
st.write("Upload an image and the model will classify it into waste categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # # Preprocess image
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown(f"### 🏷 Prediction: **{predicted_class}**")
    st.markdown(f"### 📈 Confidence: **{confidence:.2f}%**")
