import os
from dotenv import load_dotenv
import streamlit as st
from modal import Function
load_dotenv()
if st.secrets.get("modal"):
    os.environ["MODAL_TOKEN_ID"]     = st.secrets["modal"]["token_id"]
    os.environ["MODAL_TOKEN_SECRET"] = st.secrets["modal"]["token_secret"]

APP_NAME  = "cat-dog-app"
FUNC_NAME = "classify_image"
classify = Function.from_name(APP_NAME, FUNC_NAME)

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 vs 🐶 Classifier")

uploaded = st.file_uploader("Upload a cat or dog photo", type=["png","jpg","jpeg"])
if not uploaded:
    st.stop()

st.image(uploaded, use_container_width=True, caption="Your upload")
img_bytes = uploaded.read()

with st.spinner("Identifying…"):
    invocation = classify.remote(img_bytes)
    result     = invocation.result()

cats_pct = result["all_scores"]["cats"] * 100
dogs_pct = result["all_scores"]["dogs"] * 100

st.write(f"🐱 Cat: {cats_pct*100:}%   🐶 Dog: {dogs_pct*100:}%")
if result["predicted_label"] == "cats":
    st.success("It's a **CAT**!")
else:
    st.success("It's a **DOG**!")
