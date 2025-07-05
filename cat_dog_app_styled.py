import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import requests
import io

# ✅ Load the model from GitHub
@st.cache_resource
def load_model_from_github():
    url = "https://raw.githubusercontent.com/Harshit1983/cat-dog-classifier/main/svm_hog_cat_dog_model.pkl"
    response = requests.get(url)
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model_from_github()
st.success("✅ Model loaded successfully.")

# 🌸 Custom UI Style
st.markdown("""
    <style>
        .reportview-container {
            background-color: #fdf6f0;
        }
        h1 {
            color: #ff4b4b;
        }
        .stButton>button {
            background-color: #ff9f9f;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 🐾 App Title
st.title("🐾 Cat vs Dog Classifier")
st.subheader("Upload an image to know if it's a 🐱 Cat or 🐶 Dog")

# 📷 Upload file
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧠 Preprocessing
    image_resized = resize(np.array(image), (64, 64))
    gray_image = rgb2gray(image_resized)
    features, _ = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    features = features.reshape(1, -1)

    # 🤖 Prediction
    prediction = model.predict(features)[0]
    confidence = model.decision_function(features)[0]
    margin = abs(confidence)

    if margin < 0.2:
        st.warning("⚠️ Hmm... this image doesn't look like a cat or dog.\nAre you trying to test Shree? 😅")
        st.info("Tip: Please upload a **clear cat or dog image** for accurate prediction.")
    else:
        if prediction == 0:
            st.success("😺 It's a **Cat**!")
        elif prediction == 1:
            st.success("🐶 It's a **Dog**!")
