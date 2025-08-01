import os
from dotenv import load_dotenv
import streamlit as st
from modal import Function

load_dotenv()
APP_NAME, FUNC_NAME = "cat-dog-app", "classify_image"
classify = Function.from_name(APP_NAME, FUNC_NAME)

st.set_page_config(page_title="Cat vs Dog", layout="centered")
st.title("🐱 vs 🐶 Classifier")

uploaded = st.file_uploader("Upload a photo", type=["png","jpg","jpeg"])
if not uploaded:
    st.stop()

st.image(uploaded, use_container_width=True, caption="Your upload")
img_bytes = uploaded.read()

with st.spinner("Identifying…"):
    result = classify.remote(img_bytes)

st.write(f"🐱 Cat: {result['cats_prob']:.1f}%   🐶 Dog: {result['dogs_prob']:.1f}%")
st.success(f"It’s a **{result['label'].upper()}**!")
