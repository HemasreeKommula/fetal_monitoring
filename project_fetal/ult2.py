import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# --- TITLE ---
st.title("🤖 AI-Powered Ultrasound Diagnosis")
st.subheader("Fetal Health and Condition Detection")

# --- SIDEBAR INFO ---
st.sidebar.header("Project Info")
st.sidebar.markdown("""
This app uses a trained deep learning model to analyze ultrasound images and classify fetal health conditions such as:
- **Normal**
- **Malnutrition**
- **Heart Abnormalities**
- **Other Anomalies**
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    model = load_model("fetal_health_model.h5")
    return model

model = load_trained_model()
st.write("✅ Model expects input shape:", model.input_shape)

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("📤 Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="🖼 Uploaded Ultrasound Image", use_column_width=True)

    # --- PREPROCESSING ---
    img_size = (128, 128)  # ✅ Correct size for your model
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 128, 128, 1)

    # --- PREDICTION ---
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_names = ["Normal", "Malnutrition", "Heart Abnormality", "Other Anomaly"]
    result = class_names[predicted_class]

    st.success(f"🔍 **Predicted Condition:** {result}")

    # --- HEALTH SUGGESTIONS + PRECAUTIONS ---
    if result == "Normal":
        st.info("✅ Condition appears normal. Regular monitoring is advised.")
        st.markdown("""
        **Precautions:**
        - Maintain regular prenatal checkups.
        - Ensure balanced maternal diet and hydration.
        - Avoid harmful substances (alcohol, tobacco, unprescribed meds).
        """)

    elif result == "Malnutrition":
        st.warning("⚠️ Signs of fetal malnutrition detected. Recommend nutritional assessment.")
        st.markdown("""
        **Precautions for Malnutrition:**
        - Increase maternal intake of iron, folic acid, and protein.
        - Monitor fetal growth through regular ultrasounds.
        - Avoid fasting and ensure small, frequent meals.
        - Consult a maternal nutritionist for a tailored diet plan.
        """)

    elif result == "Heart Abnormality":
        st.error("❤️ Possible heart issue detected. Suggest fetal echocardiography.")
        st.markdown("""
        **Precautions for Heart Abnormality:**
        - Schedule a **fetal echocardiogram** with a pediatric cardiologist.
        - Avoid stress, and manage maternal blood pressure and glucose levels.
        - Consider genetic counseling for further screening.
        - Plan delivery in a hospital with NICU and cardiac surgery facilities.
        """)

    else:
        st.error("❗ Unusual anomaly found. Recommend further medical evaluation.")
        st.markdown("""
        **Precautions for Other Anomalies:**
        - Perform advanced imaging (MRI or 3D ultrasound) if needed.
        - Consult with a fetal medicine specialist.
        - Prepare for possible interventions or early delivery planning.
        - Ensure hospital readiness with pediatric specialists during delivery.
        """)

else:
    st.info("Please upload an ultrasound image to get started.")
