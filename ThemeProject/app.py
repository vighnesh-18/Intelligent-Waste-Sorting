import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import os

# Constants
IMAGE_SIZE = 32
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
dustbin_colors = {
    'cardboard': 'yellow',
    'glass': 'green',
    'metal': 'yellow',
    'paper': 'blue',
    'plastic': 'red',
    'trash': 'red'
}
CSV_FILE = "predictions_history.csv"

# Load model
model = tf.keras.models.load_model('model.h5')

# Preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    return image

# Predict function
def classify_image(image):
    image = preprocess_image(image)
    y_pred = model.predict(image)
    idx = np.argmax(y_pred, axis=1)[0]
    conf = y_pred[0][idx]
    label = class_names[idx]
    return label, conf

# Ensure CSV file exists and create if not
if not os.path.exists(CSV_FILE):
    # Create the CSV file with header if it doesn't exist
    df = pd.DataFrame(columns=["Label", "Confidence", "Dustbin Color"])
    df.to_csv(CSV_FILE, index=False)

# Sidebar
st.sidebar.title("♻️ Waste Classifier App")
st.sidebar.info("Upload images of waste and get their classification.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Raviteja | Vighnesh")

# Main UI
st.title("🗑️ Smart Waste Classifier")
st.subheader("Upload images to identify their waste categories")

# File uploader (multiple images allowed)
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Option to clear the CSV file
if st.sidebar.button("Clear Prediction History"):
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        # Re-create the CSV with the correct headers after deletion
        df = pd.DataFrame(columns=["Label", "Confidence", "Dustbin Color"])
        df.to_csv(CSV_FILE, index=False)
        st.sidebar.success("Prediction history cleared successfully!")

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        st.write("")
        
        # Prediction for each image
        label, conf = classify_image(image)
        
        st.subheader("Prediction:")
        st.success(f"🧾 **{label.upper()}** with **{conf*100:.2f}%** confidence")
        
        st.progress(int(conf * 100))
        
        # Show matching sample (optional: replace with real examples)
        label_images = {
            'cardboard': 'yellow.png',
            'glass': 'green.png',
            'metal': 'yellow.png',
            'paper': 'blue.png',
            'plastic': 'red.png',
            'trash': 'red.png'
        }
        sample_image = label_images.get(label, 'default.png')
        st.markdown("### Example of this category:")
        st.image(f"utils/{sample_image}", width=300, caption=f"Sample: {label}")
        
        # Determine dustbin color based on the label
        dustbin_color = dustbin_colors.get(label, 'gray')  # default to gray if no match
        
        # Save the prediction to the CSV file
        prediction_data = pd.DataFrame([[label, conf*100, dustbin_color]], columns=["Label", "Confidence", "Dustbin Color"])
        prediction_data.to_csv(CSV_FILE, mode='a', header=False, index=False)

    # Downloadable prediction history as CSV
    st.download_button(
        label="📄 Download Prediction History",
        data=open(CSV_FILE, "rb").read(),
        file_name="predictions_history.csv",
        mime="text/csv"
    )
