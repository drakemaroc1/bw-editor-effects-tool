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


# Lazy load API keys - will be set on first use
_fal_key_loaded = False
_gemini_client = None

def ensure_fal_key():
    global _fal_key_loaded
    if not _fal_key_loaded:
        try:
            os.environ['FAL_KEY'] = load_fal_api_key()
            _fal_key_loaded = True
        except FileNotFoundError:
            pass  # Will fail later when actually used

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            _gemini_client = load_gemini_client()
        except FileNotFoundError:
            pass
    return _gemini_client


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
    Expand image to 9:16 vertical using Nano Banana Pro with explicit outpainting.
    Generates new content above/below to fill the vertical frame.
    Output: 4K resolution at 9:16 aspect ratio.
    """
    ensure_fal_key()
    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "image_urls": [image_url],
            "prompt": "Expand this professional real estate photo to vertical 9:16 format. Generate natural content above (sky or ceiling) and below (ground or floor) to fill the frame. Keep the original image content centered and unchanged.",
            "aspect_ratio": "9:16",
            "resolution": "4K",
            "output_format": "png"
        }
    )
    return result["images"][0]["url"]


def outpaint_to_vertical_legacy(image_url: str) -> str:
    """
    LEGACY: Old outpaint method that creates visible seams.
    Kept for comparison/testing. DO NOT USE IN PRODUCTION.
    """
    ensure_fal_key()
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
    response = get_gemini_client().models.generate_content(
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
    Transform image using fal.ai Nano Banana Pro /edit.
    Creates start frames for Seedance reveal effects.
    """
    ensure_fal_key()
    prompts = {
        "reno": "This same building completely stripped down to bare construction studs and exposed framing, active demolition renovation site, no furniture just raw structure.",
        "staging_inside": "Make this room bare and remove all furniture. Empty room with no furniture or decor.",
        "staging_outside": "Make this outside blank, no grass plants or trees, keep home exact same. Flat dirt texture, no landscaping.",
        "3d_price": "Add 3D serif text integrated into the scene, reflecting the environment: {text}",
        "3d_city": "Add 3D serif text integrated into the scene, reflecting the environment: {text}",
        "3d_beds": "Add 3D serif text integrated into the scene, reflecting the environment: {text}",
    }
    
    prompt = prompts.get(effect)
    if not prompt:
        return image_url
    
    # Replace {text} placeholder with actual text content for 3D text effects
    if text_content and "{text}" in prompt:
        prompt = prompt.replace("{text}", text_content)
    
    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro/edit",
        arguments={
            "image_urls": [image_url],
            "prompt": prompt,
            "aspect_ratio": "9:16",  # Native 9:16 - no outpaint needed
            "resolution": "4K",
            "output_format": "png"
        }
    )
    return result["images"][0]["url"]


def transform_image(image_url: str, effect: str, text_content: str = None, use_fal: bool = True) -> str:
    """
    Transform image using Nano Banana Pro (default) or Gemini 2.5 Flash.
    
    Options:
    - use_fal=True (default): fal.ai Nano Banana Pro 4K (Gemini 3 Pro, best quality)
    - use_fal=False: Gemini 2.5 Flash Image (faster, cheaper ~$0.039)
    """
    if use_fal:
        return transform_image_fal(image_url, effect, text_content)
    return transform_image_gemini(image_url, effect, text_content)


def generate_video_seedance(first_frame_url: str, last_frame_url: str, prompt: str, duration: str = "5") -> str:
    """
    Generate video from start/end frames using Seedance 1.5 Pro.
    Use for effects with distinct start and end frames (Float, Reno).
    """
    ensure_fal_key()
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


def generate_video_veo(image_url: str, prompt: str, duration: str = "4s", end_image_url: str = None) -> str:
    """
    Generate video using VEO 3.1 Fast.
    - Single frame: uses image-to-video endpoint
    - Start+end frames: uses first-last-frame-to-video endpoint
    """
    ensure_fal_key()
    if end_image_url and end_image_url != image_url:
        # Use first/last frame endpoint for start→end transitions
        result = fal_client.subscribe(
            "fal-ai/veo3.1/fast/first-last-frame-to-video",
            arguments={
                "first_frame_url": image_url,
                "last_frame_url": end_image_url,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": "9:16",
                "generate_audio": False
            }
        )
    else:
        # Single frame endpoint
        result = fal_client.subscribe(
            "fal-ai/veo3.1/fast/image-to-video",
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": "9:16",
                "generate_audio": False
            }
        )
    return result["video"]["url"]


def generate_video(first_frame_url: str, last_frame_url: str, prompt: str, duration: str = "5", model: str = "veo") -> str:
    """
    Generate video - routes to appropriate model.
    All effects use VEO 3.1 Fast. Supports start/end frames for transitions.
    """
    if model == "veo":
        dur = duration if 's' in duration else f"{duration}s"
        # Pass end frame if different from start (for staging/reno transitions)
        end_url = last_frame_url if last_frame_url != first_frame_url else None
        return generate_video_veo(first_frame_url, prompt, dur, end_image_url=end_url)
    else:
        return generate_video_seedance(first_frame_url, last_frame_url, prompt, duration)


# Video prompts for each effect
EFFECT_PROMPTS = {
    "veo_cam": "Slow, smooth camera move forward, professional real estate video.",
    "float": "All furniture gently bobs and floats in the air with smooth, constant movement. Slow, smooth camera move forward.",
    "day_to_night": "Dramatic transition from bright daytime to cozy nighttime, sun setting, interior lights slowly turning on and glowing warm, evening atmosphere. Camera push in.",
    "staging_inside": "Furniture and decorations appear from the ground, luxury staging reveal. Slow, smooth camera move.",
    "staging_outside": "Landscaping, grass, plants, trees and outdoor furniture appear from the ground, curb appeal reveal. Slow, smooth camera move.",
    "reno": "Construction time lapse, finished room transforms revealing bare construction studs and exposed framing underneath. Slow, smooth camera move.",
    "3d_price": "One continuous shot camera fly in smooth forwards to through the opening in the 3D text. The massive 3D text is in the middle of the static environment.",
    "3d_city": "One continuous shot camera fly in smooth forwards to through the opening in the 3D text. The massive 3D text is in the middle of the static environment.",
    "3d_beds": "One continuous shot camera fly in smooth forwards to through the opening in the 3D text. The massive 3D text is in the middle of the static environment.",
}

# Which video model to use for each effect
EFFECT_VIDEO_MODEL = {
    "veo_cam": "veo",
    "float": "veo",
    "day_to_night": "veo",
    "staging_inside": "veo",
    "staging_outside": "veo",
    "reno": "veo",
    "3d_price": "veo",
    "3d_city": "veo",
    "3d_beds": "veo",
}

# Effects that need image transform (creates end frame for Seedance OR adds text for 3D)
TRANSFORM_EFFECTS = ["reno", "staging_inside", "staging_outside", "3d_price", "3d_city", "3d_beds"]

# Effects that need 9:16 conversion (using Nano Banana Pro native aspect ratio)
# ALL effects except 3D text need 9:16 input frames for best quality
NEEDS_OUTPAINT = ["veo_cam", "float", "day_to_night", "staging_inside", "staging_outside", "reno", "3d_price", "3d_city", "3d_beds"]

# 3D text effects - get outpaint + transform (Nano Banana adds text), then VEO does camera movement
TEXT_EFFECTS = ["3d_price", "3d_city", "3d_beds"]


def generate_video_kling(image_url: str, prompt: str, duration: str = "5") -> str:
    """
    Generate video using Kling 2.1 Standard on fal.ai.
    Simple I2V for Shot Generator - no transforms, just animate the image.

    Args:
        image_url: Source image URL or base64 data URL
        prompt: Motion/camera prompt
        duration: "5" or "10" seconds

    Returns:
        Video URL
    """
    ensure_fal_key()

    # Kling accepts "5" or "10" for duration
    dur = duration.replace("s", "") if isinstance(duration, str) else str(duration)
    if dur not in ["5", "10"]:
        dur = "5"

    result = fal_client.subscribe(
        "fal-ai/kling-video/v2.1/standard/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "duration": dur,
            "aspect_ratio": "9:16"
        }
    )
    return result["video"]["url"]


def load_kie_api_key() -> str:
    """Load KIE API key from config file."""
    key_path = Path.home() / ".clawdbot" / "kie_api_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    if os.environ.get('KIE_API_KEY'):
        return os.environ['KIE_API_KEY']
    raise FileNotFoundError("KIE API key not found")


def compress_image_for_kie(image_url: str, max_size_kb: int = 700) -> str:
    """
    Download, compress, and re-upload image to meet KIE API size limits.
    KIE requires images under 700KB AND a real HTTP URL (not data URLs).
    
    Returns new HTTP URL of compressed image.
    """
    from PIL import Image
    from io import BytesIO
    import tempfile
    
    print(f"[KIE Compress] Input URL type: {'data URL' if image_url.startswith('data:') else 'HTTP'}")
    print(f"[KIE Compress] URL length: {len(image_url)}")
    
    # Handle both data URLs and http URLs
    if image_url.startswith('data:'):
        # Base64 data URL - decode it
        try:
            header, encoded = image_url.split(',', 1)
            original_data = base64.b64decode(encoded)
            print(f"[KIE Compress] Decoded base64: {len(original_data)} bytes")
        except Exception as e:
            print(f"[KIE Compress] ERROR decoding base64: {e}")
            raise
    else:
        # HTTP URL - download it
        print(f"[KIE Compress] Downloading from: {image_url[:100]}...")
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            original_data = resp.content
            print(f"[KIE Compress] Downloaded: {len(original_data)} bytes")
        except Exception as e:
            print(f"[KIE Compress] ERROR downloading: {e}")
            raise
    
    max_bytes = max_size_kb * 1024
    
    # Open image
    img = Image.open(BytesIO(original_data))
    
    # Convert to RGB if needed (for JPEG)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    # Always compress and upload to get HTTP URL (KIE doesn't accept data URLs)
    max_dims = [1280, 1024, 800, 640, 480]
    qualities = [85, 75, 65, 55, 45, 35, 25]
    
    compressed_data = None
    for max_dim in max_dims:
        # Resize if needed
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            resized = img
        
        # Try different quality levels
        for quality in qualities:
            buffer = BytesIO()
            resized.save(buffer, format='JPEG', quality=quality, optimize=True)
            if buffer.tell() < max_bytes:
                buffer.seek(0)
                compressed_data = buffer.read()
                break
        if compressed_data:
            break
    
    # Fallback if nothing worked
    if not compressed_data:
        ratio = 400 / max(img.size)
        tiny = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        tiny.save(buffer, format='JPEG', quality=20, optimize=True)
        buffer.seek(0)
        compressed_data = buffer.read()
    
    # Upload to fal.ai to get a real HTTP URL (not data URL)
    print(f"[KIE Compress] Compressed to {len(compressed_data)} bytes ({len(compressed_data)/1024:.1f}KB)")
    ensure_fal_key()
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(compressed_data)
        temp_path = f.name
    
    try:
        print(f"[KIE Compress] Uploading to fal.ai...")
        http_url = fal_client.upload_file(temp_path)
        print(f"[KIE Compress] Got HTTP URL: {http_url[:80]}...")
    finally:
        os.unlink(temp_path)
    
    return http_url


def generate_video_kie(image_url: str, prompt: str, aspect_ratio: str = "9:16") -> str:
    """
    Generate video using VEO 3.1 Fast via KIE API (kie.ai).
    
    This is Drake's preferred provider for Shot Generator - 25% of Google's pricing.
    
    Args:
        image_url: Source image URL (must be publicly accessible)
        prompt: Motion/camera prompt describing how to animate the image
        aspect_ratio: "9:16" (portrait) or "16:9" (landscape)
    
    Returns:
        Video URL
    """
    import time
    
    # Compress image if needed (KIE has size limits)
    image_url = compress_image_for_kie(image_url)
    
    kie_key = load_kie_api_key()
    headers = {
        "Authorization": f"Bearer {kie_key}",
        "Content-Type": "application/json"
    }
    
    # Submit generation job
    submit_url = "https://api.kie.ai/api/v1/veo/generate"
    payload = {
        "prompt": prompt,
        "imageUrls": [image_url],
        "model": "veo3_fast",
        "generationType": "FIRST_AND_LAST_FRAMES_2_VIDEO",
        "aspect_ratio": aspect_ratio
    }
    
    resp = requests.post(submit_url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get("code") != 200:
        raise Exception(f"KIE API error: {data.get('msg', 'Unknown error')}")
    
    task_id = data["data"]["taskId"]
    print(f"[KIE] Task submitted: {task_id}")
    
    # Poll for completion (endpoint from docs: record-info, NOT detail)
    detail_url = f"https://api.kie.ai/api/v1/veo/record-info?taskId={task_id}"
    max_attempts = 120  # 10 minutes max (5s intervals)
    
    # Initial delay to let task register
    time.sleep(10)
    
    for attempt in range(max_attempts):
        print(f"[KIE] Polling attempt {attempt + 1}...")
        
        poll_resp = requests.get(detail_url, headers=headers)
        
        # Handle 404 gracefully (task might not be registered yet)
        if poll_resp.status_code == 404:
            print(f"[KIE] Got 404 - task not ready yet, waiting...")
            time.sleep(5)
            continue
        
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        
        if poll_data.get("code") != 200:
            continue  # Might be transient, keep polling
        
        success_flag = poll_data["data"].get("successFlag", 0)
        
        if success_flag == 1:  # Success
            response_data = poll_data["data"].get("response", {})
            # Try resultUrls first (documented), then videoUrl (fallback)
            result_urls = response_data.get("resultUrls", [])
            if result_urls:
                return result_urls[0]
            video_url = response_data.get("videoUrl")
            if video_url:
                return video_url
            raise Exception(f"KIE: No video URL in successful response: {response_data}")
        elif success_flag == 2:  # Failed
            raise Exception("KIE: Task failed before completion")
        elif success_flag == 3:  # Generation failed
            raise Exception("KIE: Video generation failed upstream")
        # successFlag == 0 means still generating, continue polling
    
    raise Exception("KIE: Timeout waiting for video generation")
