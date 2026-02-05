"""
BW Editor Effects Tool
Real estate video generation with parallel processing
- Phase 1a: Parallel outpaints (9:16 expansion)
- Phase 1b: Parallel transforms (Float, Reno only)
- Phase 2: Parallel video generation
"""
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_client import (
    upload_image,
    outpaint_to_vertical,
    transform_image,
    generate_video,
    EFFECT_PROMPTS,
    EFFECT_VIDEO_MODEL,
    TRANSFORM_EFFECTS,
    NEEDS_OUTPAINT,
    TEXT_EFFECTS
)

st.set_page_config(
    page_title="BW Editor Effects Tool",
    page_icon="🎬",
    layout="wide"
)

# Available effects
EFFECTS = ["VEO Cam", "Float", "Day to Night", "Staging Inside", "Staging Outside", "Reno", "3D Price", "3D City", "3D Beds"]

def get_effect_key(effect_name):
    return effect_name.lower().replace(" ", "_")

# Session state
if "videos" not in st.session_state:
    st.session_state.videos = {}
if "effect_choices" not in st.session_state:
    st.session_state.effect_choices = {}

# Title
st.title("🎬 BW Editor Effects Tool")

# Sidebar
with st.sidebar:
    st.header("Settings")
    duration = st.select_slider("Duration", options=["4s", "6s", "8s"], value="4s")
    
    st.divider()
    
    st.header("3D Text Content")
    text_price = st.text_input("Price", "$1,250,000")
    text_city = st.text_input("City", "Austin, TX")
    text_beds = st.text_input("Beds/Baths", "4 Bed / 3 Bath")
    
    st.divider()
    
    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
st.subheader("Upload Images & Assign Effects")

uploaded_files = st.file_uploader(
    "Upload listing images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    num_files = len(uploaded_files)
    
    # Initialize effect choices
    for i in range(num_files):
        if i not in st.session_state.effect_choices:
            st.session_state.effect_choices[i] = "VEO Cam"
    
    # Display grid
    cols_per_row = 4
    for row_start in range(0, num_files, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            img_idx = row_start + col_idx
            if img_idx < num_files:
                with col:
                    st.image(uploaded_files[img_idx], use_container_width=True)
                    current = st.session_state.effect_choices.get(img_idx, "VEO Cam")
                    idx = EFFECTS.index(current) if current in EFFECTS else 0
                    new_effect = st.selectbox(
                        f"Effect {img_idx + 1}",
                        EFFECTS,
                        index=idx,
                        key=f"effect_{img_idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.effect_choices[img_idx] = new_effect
    
    st.divider()
    
    # Generate button
    if st.button("🚀 Generate Videos", type="primary", use_container_width=True):
        st.session_state.videos = {}
        progress = st.progress(0)
        status = st.empty()
        
        # Prepare all jobs with their data
        jobs = []
        for i, file in enumerate(uploaded_files):
            effect_name = st.session_state.effect_choices[i]
            effect_key = get_effect_key(effect_name)
            image_bytes = file.read()
            file.seek(0)
            
            # Get text content for 3D text effects
            text_content = None
            if effect_key == "3d_price":
                text_content = text_price
            elif effect_key == "3d_city":
                text_content = text_city
            elif effect_key == "3d_beds":
                text_content = text_beds
            
            jobs.append({
                "idx": i,
                "effect_name": effect_name,
                "effect_key": effect_key,
                "image_bytes": image_bytes,
                "filename": file.name,
                "text_content": text_content,
                "needs_outpaint": effect_key in NEEDS_OUTPAINT,
                "needs_transform": effect_key in TRANSFORM_EFFECTS,
                "is_text_effect": effect_key in TEXT_EFFECTS,
                "video_model": EFFECT_VIDEO_MODEL.get(effect_key, "veo"),
                "prompt": EFFECT_PROMPTS.get(effect_key, EFFECT_PROMPTS["veo_cam"])
            })
        
        total_steps = num_files * 3  # upload + outpaint/skip + video
        completed_steps = 0
        
        # === PHASE 1a: Parallel uploads ===
        status.text(f"[Phase 1a] Uploading {num_files} images...")
        
        def upload_job(job):
            url = upload_image(job["image_bytes"], job["filename"])
            return {"idx": job["idx"], "url": url}
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(upload_job, j): j for j in jobs}
            for future in as_completed(futures):
                result = future.result()
                jobs[result["idx"]]["image_url"] = result["url"]
                completed_steps += 1
                progress.progress(completed_steps / total_steps)
        
        # === PHASE 1b: Parallel outpaints (only for effects that need it) ===
        outpaint_jobs = [j for j in jobs if j["needs_outpaint"]]
        text_jobs = [j for j in jobs if j["is_text_effect"]]
        
        if outpaint_jobs:
            status.text(f"[Phase 1b] Expanding {len(outpaint_jobs)} images to 9:16...")
            
            def outpaint_job(job):
                vertical = outpaint_to_vertical(job["image_url"])
                return {"idx": job["idx"], "vertical": vertical}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(outpaint_job, j): j for j in outpaint_jobs}
                for future in as_completed(futures):
                    result = future.result()
                    jobs[result["idx"]]["vertical_url"] = result["vertical"]
                    completed_steps += 1
                    progress.progress(completed_steps / total_steps)
        
        # Text effects skip outpaint - use original image
        for job in text_jobs:
            job["vertical_url"] = job["image_url"]
            completed_steps += 1
            progress.progress(completed_steps / total_steps)
        
        # === PHASE 1c: Parallel transforms (only Float, Reno) ===
        transform_jobs = [j for j in jobs if j["needs_transform"]]
        
        if transform_jobs:
            status.text(f"[Phase 1c] Transforming {len(transform_jobs)} images...")
            
            def transform_job(job):
                transformed = transform_image(job["vertical_url"], job["effect_key"], job.get("text_content"))
                return {"idx": job["idx"], "transformed": transformed}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(transform_job, j): j for j in transform_jobs}
                for future in as_completed(futures):
                    result = future.result()
                    jobs[result["idx"]]["end_frame"] = result["transformed"]
        
        # Set start/end frames for all jobs
        for job in jobs:
            if job["is_text_effect"]:
                # 3D text: original image only, VEO adds text in video
                job["start_frame"] = job["image_url"]
                job["end_frame"] = job["image_url"]
            elif job["needs_transform"]:
                # Float/Reno: vertical start, transformed end
                job["start_frame"] = job["vertical_url"]
                # end_frame already set by transform
            else:
                # VEO Cam, Day→Night, Staging: vertical frame only
                job["start_frame"] = job["vertical_url"]
                job["end_frame"] = job["vertical_url"]
        
        # === PHASE 2: Parallel video generation ===
        status.text(f"[Phase 2] Generating {num_files} videos in parallel...")
        
        def generate_video_job(job):
            try:
                video_url = generate_video(
                    job["start_frame"],
                    job.get("end_frame", job["start_frame"]),
                    job["prompt"],
                    duration,
                    model=job["video_model"]
                )
                return {
                    "idx": job["idx"],
                    "status": "success",
                    "effect": job["effect_name"],
                    "video_url": video_url,
                    "start_frame": job["start_frame"],
                    "end_frame": job.get("end_frame", job["start_frame"])
                }
            except Exception as e:
                return {
                    "idx": job["idx"],
                    "status": "error",
                    "effect": job["effect_name"],
                    "error": str(e)
                }
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(generate_video_job, j): j for j in jobs}
            for future in as_completed(futures):
                result = future.result()
                st.session_state.videos[result["idx"]] = result
                completed_steps += 1
                progress.progress(completed_steps / total_steps)
        
        status.text("✅ Complete!")
        st.rerun()
    
    # Display results
    if st.session_state.videos:
        st.divider()
        st.subheader("Generated Videos")
        
        for i in range(num_files):
            if i not in st.session_state.videos:
                continue
            
            result = st.session_state.videos[i]
            
            with st.expander(f"Video {i+1}: {result['effect']}", expanded=True):
                if result["status"] == "success":
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.caption("Start Frame")
                        st.image(result.get("start_frame", ""), use_container_width=True)
                    with col2:
                        st.caption("Video")
                        st.video(result["video_url"])
                    with col3:
                        st.caption("End Frame")
                        if result.get("end_frame") != result.get("start_frame"):
                            st.image(result.get("end_frame", ""), use_container_width=True)
                        else:
                            st.info("Same as start")
                        st.download_button(
                            "⬇️ Download",
                            data=requests.get(result["video_url"]).content,
                            file_name=f"video_{i+1}.mp4",
                            mime="video/mp4",
                            key=f"dl_{i}"
                        )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")

else:
    st.info("Upload listing images to get started")
    
    with st.expander("Available Effects"):
        st.markdown("""
        **VEO Cam** — Camera motion only (outpaint → VEO)
        
        **Float** — Furniture floats up (outpaint → transform → Seedance)
        
        **Day to Night** — Day→night transition (outpaint → VEO)
        
        **Staging Inside** — Interior staging (outpaint → VEO)
        
        **Staging Outside** — Exterior staging (outpaint → VEO)
        
        **Reno** — Construction reveal (outpaint → transform → Seedance)
        
        **3D Price/City/Beds** — Text overlay (original → VEO, no outpaint)
        """)
