"""
Nano Banana Pro Direct Tester
JUST the model output - no outpaint, no video, no bullshit
"""
import streamlit as st
from google import genai
from google.genai import types
import base64
import os
from pathlib import Path

st.set_page_config(page_title="NB Pro Tester", page_icon="🍌", layout="wide")

# Init Gemini
@st.cache_resource
def get_client():
    sa_path = Path.home() / ".clawdbot" / "vertex-ai-service-account.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(sa_path)
    return genai.Client(vertexai=True, project="free-trial-flow-470815", location="us-central1")

client = get_client()

st.title("🍌 Nano Banana Pro - Direct Test")
st.caption("Raw model output only. No outpaint. No video.")

# Sidebar
with st.sidebar:
    st.header("Model")
    model = st.radio(
        "Select",
        ["gemini-3-pro-image-preview", "gemini-2.5-flash-image"],
        index=1  # Default to 2.5 since 3-pro might not be available
    )
    st.code(model)
    
    st.divider()
    
    aspect = st.radio("Aspect Ratio", ["9:16", "16:9", "1:1", "auto"], index=0)
    
    st.divider()
    
    st.markdown("**Presets**")
    if st.button("Float", use_container_width=True):
        st.session_state.prompt = "Edit: Make all furniture float and levitate in the air"
    if st.button("Reno", use_container_width=True):
        st.session_state.prompt = "Edit: Strip to bare construction studs, exposed framing, no furniture"
    if st.button("Night", use_container_width=True):
        st.session_state.prompt = "Edit: Transform to nighttime with warm glowing interior lights"
    if st.button("Stage", use_container_width=True):
        st.session_state.prompt = "Edit: Add elegant modern furniture to stage this room"

# Init session
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "result" not in st.session_state:
    st.session_state.result = None

# Main
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, use_container_width=True)
    
    prompt = st.text_area("Prompt", value=st.session_state.prompt, height=100)
    st.session_state.prompt = prompt
    
    c1, c2 = st.columns(2)
    with c1:
        go = st.button("🚀 Generate", type="primary", use_container_width=True, disabled=not uploaded or not prompt)
    with c2:
        retry = st.button("🔄 Retry", use_container_width=True, disabled=not st.session_state.result)

with col2:
    st.subheader(f"Output ({model})")
    
    if go or retry:
        with st.spinner(f"Calling {model}..."):
            try:
                # Get image bytes
                if go:
                    img_bytes = uploaded.read()
                    uploaded.seek(0)
                    st.session_state.input_bytes = img_bytes
                else:
                    img_bytes = st.session_state.input_bytes
                
                # Build request
                image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                
                # Add aspect ratio to prompt
                full_prompt = f"{prompt}\n\nOutput aspect ratio: {aspect}"
                text_part = types.Part(text=full_prompt)
                
                # Call API
                response = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[image_part, text_part])],
                    config=types.GenerateContentConfig(response_modalities=["image", "text"])
                )
                
                # Extract result
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        b64 = base64.b64encode(part.inline_data.data).decode()
                        st.session_state.result = f"data:{part.inline_data.mime_type};base64,{b64}"
                        st.session_state.result_size = len(part.inline_data.data)
                    elif part.text:
                        st.session_state.result_text = part.text
                        
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.result:
        st.image(st.session_state.result, use_container_width=True)
        st.caption(f"Size: {st.session_state.get('result_size', 0) / 1024:.1f} KB")
        if st.session_state.get('result_text'):
            st.text(st.session_state.result_text)
