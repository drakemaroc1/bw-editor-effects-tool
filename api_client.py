"""
API client for Editor Effects Tool
Handles Nano Banana Pro (fal.ai) and VEO 3.1 Fast (Vertex AI) calls

For Streamlit Cloud deployment, uses st.secrets for API keys
"""

import os
import base64
import tempfile
import time

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

import fal_client

# VEO config
VEO_MODEL = "veo-3.1-fast-generate-001"  # Production model
VEO_LOCATION = "us-central1"


def load_fal_key():
    """Load fal.ai API key from secrets or env"""
    if HAS_STREAMLIT and hasattr(st, 'secrets') and 'FAL_KEY' in st.secrets:
        os.environ['FAL_KEY'] = st.secrets['FAL_KEY']
        return True
    elif os.environ.get('FAL_KEY'):
        return True
    else:
        # Try local file fallback
        from pathlib import Path
        key_path = Path.home() / ".clawdbot" / "fal_api_key.txt"
        if key_path.exists():
            os.environ['FAL_KEY'] = key_path.read_text().strip()
            return True
    return False


def get_veo_client():
    """Initialize Google GenAI client"""
    try:
        from google import genai
        import json
        
        # Try Streamlit secrets first
        if HAS_STREAMLIT and hasattr(st, 'secrets') and 'GOOGLE_SERVICE_ACCOUNT' in st.secrets:
            # Write service account to temp file
            sa_data = dict(st.secrets['GOOGLE_SERVICE_ACCOUNT'])
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sa_data, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
            project_id = sa_data.get('project_id')
        else:
            # Try local file
            from pathlib import Path
            key_path = Path.home() / ".clawdbot" / "vertex-ai-service-account.json"
            if key_path.exists():
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(key_path)
                with open(key_path) as f:
                    sa_info = json.load(f)
                    project_id = sa_info.get('project_id')
            else:
                return None
        
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=VEO_LOCATION
        )
        return client
    except ImportError:
        print("google-genai not installed. Run: pip install google-genai")
        return None
    except Exception as e:
        print(f"VEO client error: {e}")
        return None


# Initialize on import
load_fal_key()


def generate_nano_banana_image(
    prompt: str,
    aspect_ratio: str = "9:16",
    num_images: int = 1
) -> dict:
    """Generate image using Nano Banana Pro via fal.ai"""
    result = fal_client.subscribe(
        "fal-ai/nano-banana-pro",
        arguments={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images
        }
    )
    return result


def generate_effect_frames(
    property_description: str,
    effect_type: str,
    text_content: str = None
) -> dict:
    """Generate start and end frames for an effect"""
    
    prompts = get_effect_prompts(property_description, effect_type, text_content)
    
    # Generate start frame
    start_result = generate_nano_banana_image(prompts['start'])
    
    # Generate end frame (if needed)
    end_result = None
    if prompts.get('end'):
        end_result = generate_nano_banana_image(prompts['end'])
    
    return {
        'start_frame': start_result['images'][0]['url'] if start_result.get('images') else None,
        'end_frame': end_result['images'][0]['url'] if end_result and end_result.get('images') else None,
        'effect_type': effect_type
    }


def get_effect_prompts(property_desc: str, effect_type: str, text_content: str = None) -> dict:
    """Get prompts for a specific effect type"""
    
    prompts = {
        "Lighting Transform (Day→Night)": {
            "start": f"{property_desc}, exterior photograph, bright natural daylight, clear blue sky, sharp shadows, professional architectural photography, 9:16 vertical aspect ratio, high resolution",
            "end": f"{property_desc}, exterior photograph, dusk twilight hour, warm amber interior lights glowing through windows, deep blue evening sky, soft ambient lighting, professional architectural photography, 9:16 vertical aspect ratio, high resolution"
        },
        "Virtual Staging (Empty→Furnished)": {
            "start": f"{property_desc}, empty unfurnished interior, clean floors, natural window light, spacious empty room, professional interior photography, 9:16 vertical aspect ratio, high resolution",
            "end": f"{property_desc}, beautifully furnished interior, modern luxury furniture, styled decor, warm inviting atmosphere, professional interior design photography, 9:16 vertical aspect ratio, high resolution"
        },
        "Furniture Float": {
            "start": f"{property_desc}, luxury furnished interior, furniture resting on floor, styled modern decor, warm lighting, professional interior photography, 9:16 vertical aspect ratio, high resolution",
            "end": f"{property_desc}, luxury furnished interior, furniture floating 8 inches above floor, subtle magical levitation, same lighting and decor, professional interior photography with surreal element, 9:16 vertical aspect ratio, high resolution"
        },
        "Construction → Finished": {
            "start": f"{property_desc}, under construction state, exposed framing or renovation in progress, construction materials visible, work in progress, documentary photography style, 9:16 vertical aspect ratio, high resolution",
            "end": f"{property_desc}, fully completed renovation, pristine finished state, polished surfaces, move-in ready, professional architectural photography, 9:16 vertical aspect ratio, high resolution"
        },
        "Close-up Punch-in": {
            "start": f"Extreme close-up detail shot of {property_desc}, macro photography style, shallow depth of field, beautiful bokeh, intimate detail, texture visible, premium architectural detail photography, 9:16 vertical aspect ratio, high resolution",
            "end": None
        },
        "Push + 3D Text (Price)": {
            "start": f"{property_desc}, professional architectural photography, elegant floating 3D text reading \"{text_content}\" hovering above property, metallic gold/white text with subtle shadow, premium real estate marketing, 9:16 vertical aspect ratio, high resolution",
            "end": None
        },
        "Push + 3D Text (City)": {
            "start": f"{property_desc}, professional architectural photography, elegant floating 3D text reading \"{text_content}\" hovering above property, metallic gold/white text with subtle shadow, premium real estate marketing, 9:16 vertical aspect ratio, high resolution",
            "end": None
        },
        "Push + 3D Text (Beds/Baths)": {
            "start": f"{property_desc}, professional architectural photography, elegant floating 3D text reading \"{text_content}\" hovering above property, metallic gold/white text with subtle shadow, premium real estate marketing, 9:16 vertical aspect ratio, high resolution",
            "end": None
        }
    }
    
    return prompts.get(effect_type, {"start": property_desc, "end": None})


def get_camera_movement_prompt(property_desc: str, movement_type: str) -> str:
    """Get VEO prompt for camera movement"""
    
    prompts = {
        "Orbit": f"Cinematic orbit shot around {property_desc}, camera smoothly arcs 20 degrees to the right, maintaining focus on subject, architectural visualization quality, golden hour lighting, 9:16 vertical aspect ratio, 4 seconds, smooth steady motion",
        "Dolly In": f"Elegant dolly forward through {property_desc}, camera glides smoothly into the space, revealing depth and detail, cinematic entrance shot, shallow depth of field, architectural film quality, 9:16 vertical aspect ratio, 4 seconds, buttery smooth motion",
        "Crane Up": f"Dramatic crane shot rising upward revealing {property_desc}, starting low and ascending to show scale and grandeur, architectural visualization, epic reveal shot, 9:16 vertical aspect ratio, 4 seconds, smooth vertical motion",
        "Push": f"Slow cinematic push toward {property_desc}, subtle forward motion, focus maintained on subject, architectural showcase quality, 9:16 vertical aspect ratio, 4 seconds, gentle forward drift",
        "Pull/Reveal": f"Cinematic pull-back shot revealing {property_desc}, camera slowly retreats to show full scope and context, establishing shot feel, architectural grandeur, 9:16 vertical aspect ratio, 4 seconds, smooth backward glide"
    }
    
    return prompts.get(movement_type, f"Cinematic shot of {property_desc}, 9:16 vertical, 4 seconds")


def generate_video_from_image_veo(
    image_path_or_url: str,
    prompt: str,
    duration_seconds: int = 4,
    aspect_ratio: str = "9:16"
) -> dict:
    """Generate video from an image using VEO 3.1 Fast"""
    try:
        from google.genai import types
        import urllib.request
        
        client = get_veo_client()
        if not client:
            return {'error': 'VEO client not initialized. Check credentials.'}
        
        # Load image
        if image_path_or_url.startswith('http'):
            with urllib.request.urlopen(image_path_or_url) as response:
                image_bytes = response.read()
            
            suffix = '.png' if '.png' in image_path_or_url.lower() else '.jpg'
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            
            image = types.Image.from_file(location=tmp_path)
        else:
            image = types.Image.from_file(location=image_path_or_url)
        
        # Generate video
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            image=image,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation="allow_adult"
            )
        )
        
        # Poll for completion
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)
        
        if operation.response and operation.response.generated_videos:
            video = operation.response.generated_videos[0]
            return {
                'video_data': video.video.video_bytes if hasattr(video.video, 'video_bytes') else None,
                'video_gcs_uri': video.video.uri if hasattr(video.video, 'uri') else None,
                'status': 'complete'
            }
        else:
            return {'error': 'No video generated', 'status': 'failed'}
            
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}


def generate_video_from_text_veo(
    prompt: str,
    duration_seconds: int = 4,
    aspect_ratio: str = "9:16"
) -> dict:
    """Generate video from text prompt using VEO 3.1 Fast"""
    try:
        from google.genai import types
        
        client = get_veo_client()
        if not client:
            return {'error': 'VEO client not initialized. Check credentials.'}
        
        operation = client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                person_generation="allow_adult"
            )
        )
        
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)
        
        if operation.response and operation.response.generated_videos:
            video = operation.response.generated_videos[0]
            return {
                'video_data': video.video.video_bytes if hasattr(video.video, 'video_bytes') else None,
                'video_gcs_uri': video.video.uri if hasattr(video.video, 'uri') else None,
                'status': 'complete'
            }
        else:
            return {'error': 'No video generated', 'status': 'failed'}
            
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}
