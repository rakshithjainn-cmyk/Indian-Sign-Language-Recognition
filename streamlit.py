# app/streamlit_app.py
import streamlit as st
from googletrans import Translator
import json
from src.predict import predict_sequence
import os

st.set_page_config(page_title='ISL Recognizer', layout='wide')

st.title('Indian Sign Language Recognition — Multilingual Translation')

with st.sidebar:
    st.header('Settings')
    model_path = st.text_input('Model path', value='models/best_model.h5')
    label_map_path = st.text_input('Label map path', value='models/label_map.json')
    target_lang = st.selectbox('Translate to', options=['en','hi','es','fr','de','bn','ta','te','ml','mr'])

st.write('Upload sequence frames (as multiple image files) or a zip of frames in correct order.')
uploaded = st.file_uploader('Upload frames', type=['png','jpg','jpeg','zip'], accept_multiple_files=True)

if uploaded:
    # if zip provided, extract
    img_paths = []
    tmp_dir = 'tmp_upload'
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(uploaded):
        # save file
        outp = os.path.join(tmp_dir, f'{i}_{f.name}')
        with open(outp,'wb') as wf:
            wf.write(f.getbuffer())
        img_paths.append(outp)

    st.image([p for p in img_paths[:10]], width=120, caption=[os.path.basename(p) for p in img_paths[:10]])

    if st.button('Predict & Translate'):
        # load label map
        with open(label_map_path,'r') as f:
            label_map = json.load(f)
        idx, probs = predict_sequence(img_paths, model_path=model_path)
        label = label_map[idx]
        st.success(f'Predicted sign: {label} (class {idx})')
        translator = Translator()
        trans = translator.translate(label, dest=target_lang)
        st.info(f'Translation ({target_lang}): {trans.text}')

        st.write('Class probabilities:')
        for i,p in enumerate(probs):
            st.write(f"{i}: {label_map[i]} -> {p:.3f}")
