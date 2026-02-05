"""
BW Editor Effects Tool - TESTER PAGE
Compare fal.ai vs Google Direct API for Nano Banana Pro.
"""
import streamlit as st
import fal_client
import base64
import os
from pathlib import Path

st.set_page_config(page_title="Nano Banana Tester", page_icon="🧪", layout="wide")

# Load credentials
def load_fal_key():
    if os.environ.get('FAL_KEY'):
        return os.environ['FAL_KEY']
    key_path = Path.home() / ".clawdbot" / "fal_api_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    raise FileNotFoundError("FAL API key not found")

def load_gemini_client():
    """Load Google Gemini client via Vertex AI."""
    from google import genai
    sa_path = Path.home() / ".clawdbot" / "vertex-ai-service-account.json"
    if sa_path.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(sa_path)
        return genai.Client(
            vertexai=True,
            project="free-trial-flow-470815",
            location="us-central1"
        )
    raise FileNotFoundError("Google credentials not found")

os.environ['FAL_KEY'] = load_fal_key()

st.title("🧪 Nano Banana Pro Tester")
st.caption("Compare fal.ai wrapper vs Google Direct API")

# Settings
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# Preset prompts
PRESETS = {
    "Custom": "",
    "Float": "Make all furniture float and levitate high in the air, suspended magically above the floor",
    "Reno/Construction": "Transform this room into an active renovation site with bare studs and exposed framing, all furniture removed",
    "Day to Night": "Transform this daytime scene to nighttime with warm interior lighting glowing from windows",
    "Virtual Staging (Empty→Furnished)": "Add elegant modern furniture and decor to this empty room, luxury staging",
    "Declutter": "Remove clutter and mess, make the space clean and organized",
    "Season Change (Summer→Winter)": "Transform to winter scene with snow outside windows",
    "Pool Add": "Add a beautiful swimming pool to this backyard",
    "Reformat Only (no edit)": "Professional real estate photo, high quality",
}

preset = st.selectbox("Prompt Template", list(PRESETS.keys()))
default_prompt = PRESETS[preset] if preset != "Custom" else "Make all furniture float and levitate high in the air"
prompt = st.text_area("Transform prompt (edit freely)", default_prompt)

col1, col2, col3 = st.columns(3)
with col1:
    aspect = st.selectbox("Aspect Ratio", ["9:16", "16:9", "1:1", "4:3", "3:4"])
with col2:
    resolution = st.selectbox("Resolution (fal.ai)", ["4K", "2K", "1K"], index=0)
with col3:
    backend = st.selectbox("Backend", ["fal.ai (Nano Banana Pro)", "Google Direct (Gemini)"], index=0)

if uploaded and st.button("🔥 Generate", type="primary"):
    image_bytes = uploaded.read()
    uploaded.seek(0)
    mime = "image/png" if uploaded.name.lower().endswith('.png') else "image/jpeg"
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(uploaded, use_container_width=True)
    
    with col2:
        if "fal.ai" in backend:
            with st.spinner(f"Calling fal.ai Nano Banana Pro ({resolution}, {aspect})..."):
                data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
                result = fal_client.subscribe(
                    "fal-ai/nano-banana-pro/edit",
                    arguments={
                        "image_urls": [data_url],
                        "prompt": prompt,
                        "aspect_ratio": aspect,
                        "resolution": resolution
                    }
                )
                st.subheader(f"Result (fal.ai {resolution})")
                if result.get("images"):
                    img_url = result["images"][0]["url"]
                    st.image(img_url, use_container_width=True)
                    import requests
                    img_data = requests.get(img_url).content
                    st.caption(f"Size: {len(img_data)/1024:.1f} KB")
                    st.download_button("⬇️ Download", img_data, "fal_result.png", "image/png")
                else:
                    st.error("No image returned")
                    st.json(result)
        else:
            with st.spinner("Calling Google Gemini Direct (Vertex AI)..."):
                from google.genai import types
                client = load_gemini_client()
                
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
                full_prompt = f"{prompt}\n\nOutput aspect ratio: {aspect}"
                text_part = types.Part(text=full_prompt)
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",  # Latest Gemini image model
                    contents=[types.Content(role="user", parts=[image_part, text_part])],
                    config=types.GenerateContentConfig(response_modalities=["image", "text"])
                )
                
                st.subheader("Result (Google Direct)")
                result_image = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        result_image = part.inline_data.data
                        break
                
                if result_image:
                    st.image(result_image, use_container_width=True)
                    st.caption(f"Size: {len(result_image)/1024:.1f} KB")
                    st.download_button("⬇️ Download", result_image, "google_result.png", "image/png")
                else:
                    st.error("No image returned from Gemini")

st.divider()
st.markdown("""
**Backend comparison:**
- **fal.ai**: Uses `fal-ai/nano-banana-pro/edit` - supports resolution param (1K/2K/4K)
- **Google Direct**: Uses Vertex AI Gemini - may have different quality/limits
""")
