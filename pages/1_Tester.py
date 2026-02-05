"""
BW Editor Effects Tool - TESTER PAGE
Raw Nano Banana Pro output. NO expansion, NO outpaint.
Uses fal.ai nano-banana-pro with native 9:16 and 4K.
"""
import streamlit as st
import fal_client
import os
from pathlib import Path

st.set_page_config(page_title="Nano Banana Tester", page_icon="🧪", layout="wide")

def load_fal_key():
    if os.environ.get('FAL_KEY'):
        return os.environ['FAL_KEY']
    key_path = Path.home() / ".clawdbot" / "fal_api_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    raise FileNotFoundError("FAL API key not found")

os.environ['FAL_KEY'] = load_fal_key()

st.title("🧪 Nano Banana Pro Tester")
st.caption("Raw output — NO expansion, NO outpaint, native 9:16")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
prompt = st.text_area("Transform prompt", "Make all furniture float and levitate high in the air, suspended magically above the floor")

col1, col2 = st.columns(2)
with col1:
    aspect = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "4:3", "3:4"])
with col2:
    resolution = st.selectbox("Resolution", ["4K", "2K", "1K"], index=0)

if uploaded and st.button("🔥 Generate", type="primary"):
    with st.spinner("Calling Nano Banana Pro (4K, native aspect)..."):
        import base64
        image_bytes = uploaded.read()
        mime = "image/png" if uploaded.name.lower().endswith('.png') else "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        
        result = fal_client.subscribe(
            "fal-ai/nano-banana-pro/edit",
            arguments={
                "image_url": data_url,
                "prompt": prompt,
                "aspect_ratio": aspect,
                "resolution": resolution
            }
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            uploaded.seek(0)
            st.image(uploaded, use_container_width=True)
        with col2:
            st.subheader("Result (4K, native 9:16)")
            if result.get("images"):
                st.image(result["images"][0]["url"], use_container_width=True)
                import requests
                img_data = requests.get(result["images"][0]["url"]).content
                st.download_button("⬇️ Download", img_data, "nano_banana_4k.png", "image/png")
            else:
                st.error("No image returned")
                st.json(result)

st.divider()
st.caption("This tests fal-ai/nano-banana-pro/edit directly with resolution=4K and native aspect_ratio")
