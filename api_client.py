"""
FAL.AI + Google Gemini API Client for BW Editor Effects Tool

Pipeline:
- Vertical Expansion: fal-ai/nano-banana-pro/edit with aspect_ratio="9:16"
  (replaces old outpaint endpoint which created visible seams)
- Transform (Float/Reno): Gemini 2.5 Flash Image or Nano Banana Pro /edit
- Video: VEO 3.1 or Seedance 1.5 Pro

ARCHITECTURE FIX (2025-02-05):
The old fal-ai/image-apps-v2/outpaint endpoint placed original images in the
center and generated content above/below, creating visible "stitching" seams.
Now using Nano Banana Pro /edit with native 9:16 aspect_ratio for seamless
vertical expansion.
"""
import os
import base64
import requests
import fal_client
from pathlib import Path
from typing import Optional, List

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Google Gemini imports
from google import genai
from google.genai import types


def load_fal_api_key() -> str:
    """Load FAL_KEY from secrets, env, or config file."""
    if HAS_STREAMLIT and hasattr(st, 'secrets') and 'FAL_KEY' in st.secrets:
        return st.secrets['FAL_KEY']
    if os.environ.get('FAL_KEY'):
        return os.environ['FAL_KEY']
    key_path = Path.home() / ".clawdbot" / "fal_api_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    raise FileNotFoundError("FAL API key not found")


def load_gemini_client():
    """Load Google Gemini client using Vertex AI."""
    # Try Google AI API key first
    if os.environ.get('GOOGLE_API_KEY'):
        return genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
    
    # Use Vertex AI with service account
    sa_path = Path.home() / ".clawdbot" / "vertex-ai-service-account.json"
    if sa_path.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(sa_path)
        return genai.Client(
            vertexai=True,
            project="free-trial-flow-470815",
            location="us-central1"
        )
    
    raise FileNotFoundError("Google credentials not found")


# Set FAL_KEY in environment for fal_client
os.environ['FAL_KEY'] = load_fal_api_key()

# Initialize Gemini client
gemini_client = load_gemini_client()


def upload_image(image_bytes: bytes, filename: str = "image.png") -> str:
    """Convert image bytes to base64 data URL for fal.ai."""
    if filename.lower().endswith('.png'):
        mime = 'image/png'
    elif filename.lower().endswith(('.jpg', '.jpeg')):
        mime = 'image/jpeg'
    else:
        mime = 'image/png'
    
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime};base64,{b64}"


def outpaint_to_vertical(image_url: str) -> str:
    """
    Convert image to 9:16 vertical using Nano Banana Pro /edit.
    Uses native aspect_ratio parameter for seamless expansion (no visible seams).
    Output: 2K resolution at 9:16 aspect ratio.
    
    This replaces the old outpaint endpoint which created visible seams.
    """
    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "image_urls": [image_url],
            "prompt": "Expand this real estate photo to a vertical 9:16 format. Extend the scene naturally above and below while preserving the original content in the center. Keep the same lighting, style, and perspective. Photorealistic, professional real estate photography.",
            "aspect_ratio": "9:16",
            "resolution": "2K",  # Higher quality
            "output_format": "png"
        }
    )
    return result["images"][0]["url"]


def outpaint_to_vertical_legacy(image_url: str) -> str:
    """
    LEGACY: Old outpaint method that creates visible seams.
    Kept for comparison/testing. DO NOT USE IN PRODUCTION.
    """
    result = fal_client.subscribe(
        "fal-ai/image-apps-v2/outpaint",
        arguments={
            "image_url": image_url,
            "direction": "center",
            "output_size": {
                "width": 1080,
                "height": 1920
            }
        }
    )
    return result["images"][0]["url"]


def download_image_as_bytes(url: str) -> bytes:
    """Download image from URL and return bytes."""
    if url.startswith('data:'):
        # Already base64, decode it
        header, b64data = url.split(',', 1)
        return base64.b64decode(b64data)
    else:
        response = requests.get(url)
        response.raise_for_status()
        return response.content


def transform_image_gemini(image_url: str, effect: str, text_content: str = None) -> str:
    """
    Transform image using Google Gemini API directly.
    Uses Gemini 2.5 Flash Image for speed, can upgrade to 3 Pro for quality.
    Returns URL of transformed image.
    """
    prompts = {
        "float": "Edit this image: Make all the furniture and objects float and levitate high in the air, suspended magically above the floor. Keep the room exactly the same but everything should be floating in a dreamlike, gravity-defying way.",
        "reno": "Edit this image: Transform this room into an active renovation/construction site. Strip it down to bare studs and exposed framing. Remove all furniture and finishes, showing only the raw construction structure.",
    }
    
    prompt = prompts.get(effect)
    if not prompt:
        return image_url  # No transform needed
    
    # Download image and convert to Part
    image_bytes = download_image_as_bytes(image_url)
    
    # Create image part for Gemini
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png"
    )
    
    # Create text part
    text_part = types.Part(text=prompt)
    
    # Call Gemini API with image + prompt
    # Using gemini-2.5-flash-image for speed; upgrade to gemini-3-pro-image-preview for quality
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Content(
                role="user",
                parts=[image_part, text_part]
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["image", "text"],
        )
    )
    
    # Extract image from response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            # Convert to data URL
            b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
            mime = part.inline_data.mime_type or 'image/png'
            return f"data:{mime};base64,{b64}"
    
    # If no image in response, return original
    print(f"Warning: Gemini didn't return an image for {effect}")
    return image_url


def transform_image_fal(image_url: str, effect: str, text_content: str = None) -> str:
    """
    Alternative: Transform image using fal.ai Nano Banana Pro /edit.
    Higher quality than Gemini 2.5 Flash (uses Gemini 3 Pro Image).
    Cost: $0.15/image vs ~$0.039 for Gemini 2.5 Flash.
    """
    prompts = {
        "float": "Make all the furniture and objects float and levitate high in the air, suspended magically above the floor. Keep the room exactly the same but everything should be floating in a dreamlike, gravity-defying way.",
        "reno": "Transform this room into an active renovation/construction site. Strip it down to bare studs and exposed framing. Remove all furniture and finishes, showing only the raw construction structure.",
    }
    
    prompt = prompts.get(effect)
    if not prompt:
        return image_url
    
    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "image_urls": [image_url],
            "prompt": prompt,
            "aspect_ratio": "auto",  # Preserve input aspect ratio
            "resolution": "2K",
            "output_format": "png"
        }
    )
    return result["images"][0]["url"]


def transform_image(image_url: str, effect: str, text_content: str = None, use_fal: bool = False) -> str:
    """
    Transform already-vertical image.
    
    Options:
    - use_fal=False (default): Gemini 2.5 Flash Image (faster, cheaper ~$0.039)
    - use_fal=True: fal.ai Nano Banana Pro (Gemini 3 Pro, higher quality, $0.15)
    """
    if use_fal:
        return transform_image_fal(image_url, effect, text_content)
    return transform_image_gemini(image_url, effect, text_content)


def generate_video_seedance(first_frame_url: str, last_frame_url: str, prompt: str, duration: str = "5") -> str:
    """
    Generate video from start/end frames using Seedance 1.5 Pro.
    Use for effects with distinct start and end frames (Float, Reno).
    """
    dur = duration.replace("s", "") if duration else "5"
    
    result = fal_client.subscribe(
        "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
        arguments={
            "image_url": first_frame_url,
            "end_image_url": last_frame_url,
            "prompt": prompt,
            "duration": dur,
            "aspect_ratio": "9:16",
            "resolution": "1080p",
            "generate_audio": False
        }
    )
    return result["video"]["url"]


def generate_video_veo(image_url: str, prompt: str, duration: str = "4s") -> str:
    """
    Generate video from single frame using VEO 3.1.
    Use for effects where video model applies the transformation.
    """
    result = fal_client.subscribe(
        "fal-ai/veo3.1/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": "9:16",
            "generate_audio": False
        }
    )
    return result["video"]["url"]


def generate_video(first_frame_url: str, last_frame_url: str, prompt: str, duration: str = "5", model: str = "seedance") -> str:
    """
    Generate video - routes to appropriate model.
    model: "seedance" for start/end frame transitions, "veo" for single frame + motion
    """
    if model == "veo":
        return generate_video_veo(first_frame_url, prompt, duration if 's' in duration else f"{duration}s")
    else:
        return generate_video_seedance(first_frame_url, last_frame_url, prompt, duration)


# Video prompts for each effect
EFFECT_PROMPTS = {
    "veo_cam": "Cinematic fast camera fly forward with slight orbit, smooth camera arc, professional real estate video",
    "float": "Smooth cinematic transition, furniture gently floating upward, dreamlike magical atmosphere, camera pushing forward",
    "day_to_night": "Dramatic transition from day to night, interior lights warming up and glowing, cozy evening atmosphere, slow camera push forward",
    "staging_inside": "Professional interior staging reveal, elegant furniture appearing, luxury design transformation, camera pushing in",
    "staging_outside": "Beautiful landscaping transformation, manicured lawn and outdoor furniture appearing, curb appeal reveal, camera pushing in",
    "reno": "Dramatic construction reveal, renovation transformation, camera orbiting around the space",
    "3d_price": "Cinematic camera push forward with subtle movement, professional marketing video",
    "3d_city": "Cinematic camera push forward with subtle movement, professional marketing video",
    "3d_beds": "Cinematic camera push forward with subtle movement, professional marketing video",
}

# Which video model to use for each effect
EFFECT_VIDEO_MODEL = {
    "veo_cam": "veo",
    "float": "seedance",      # Seedance for start→end frame transition
    "day_to_night": "veo",    # VEO does day→night in video
    "staging_inside": "veo",  # VEO does staging in video
    "staging_outside": "veo", # VEO does staging in video
    "reno": "seedance",       # Seedance for start→end frame transition
    "3d_price": "veo",        # VEO adds text effect in video
    "3d_city": "veo",
    "3d_beds": "veo",
}

# Effects that need Gemini transform (creates different end frame for Seedance)
TRANSFORM_EFFECTS = ["float", "reno"]

# Effects that need outpaint (9:16 expansion)
# VEO handles 9:16 natively - no outpaint needed for VEO effects
# Only Seedance effects need pre-expanded frames
NEEDS_OUTPAINT = ["float", "reno"]  # Seedance needs 9:16 input frames

# 3D text effects - NO outpaint, NO transform, just pass original to VEO
TEXT_EFFECTS = ["3d_price", "3d_city", "3d_beds"]
