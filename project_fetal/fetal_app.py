import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Title
st.title("🤖 AI-Powered Ultrasound Diagnosis")
st.subheader("Fetal Health and Condition Detection")

# Sidebar info
st.sidebar.header("Project Info")
st.sidebar.markdown("This app uses a trained deep learning model to analyze ultrasound images and classify fetal health conditions such as **Normal**, **Malnutrition**, or **Heart Abnormalities**.")

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model("fetal_health_model.h5")  # Replace with your model path
    return model

model = load_trained_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Ultrasound Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)  # Assumes model input is 224x224
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ["Normal", "Malnutrition", "Heart Abnormality", "Other Anomaly"]
    result = class_names[predicted_class]

    st.success(f"🔍 Predicted Condition: **{result}**")

    # Suggestion
    if result == "Normal":
        st.info("✅ Condition appears normal. Regular monitoring is advised.")
    elif result == "Malnutrition":
        st.warning("⚠️ Signs of fetal malnutrition detected. Recommend nutritional assessment.")
    elif result == "Heart Abnormality":
        st.error("❤️ Possible heart issue detected. Suggest fetal echocardiography.")
    else:
        st.error("❗ Unusual anomaly found. Recommend further medical evaluation.")

else:
    st.info("Please upload an ultrasound image to get started.")