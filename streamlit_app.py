import os
from dotenv import load_dotenv
import streamlit as st
from app1 import classify_image_with_scores
load_dotenv()
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("Cat vs Dog Classifier")
uploaded = st.file_uploader("Upload a cat or dog photo", type=["png", "jpg", "jpeg"])
if uploaded:
    st.image(uploaded, use_container_width=True, caption="Your upload")
    img_bytes = uploaded.read()

    with st.spinner("Identifying…"):
        result = classify_image_with_scores(img_bytes)
    scores = result["all_scores"]
    cats_score = scores["cats"]
    dogs_score = scores["dogs"]
    st.write(f"🐱 Cat: {cats_score:}%   🐶 Dog: {dogs_score:}%")

    if result["predicted_label"] == "cats":
        st.success("🐱 It's a CAT :)")
    else:
        st.success("🐶 It's a DOG")
