import streamlit as st
from src.predict import predict_image
from app.translate import translate_text

st.title("Indian Sign Language Recognition & Translation")

uploaded_file = st.file_uploader("Upload a sign image", type=["jpg", "png"])
language = st.selectbox("Translate to", ["en", "fr", "es", "hi"])

if uploaded_file is not None:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    prediction = predict_image("temp.png")
    translation = translate_text(prediction, dest=language)
    st.image("temp.png")
    st.write(f"**Detected Sign:** {prediction}")
    st.write(f"**Translation:** {translation}")
