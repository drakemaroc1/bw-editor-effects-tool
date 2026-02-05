"""
BW Editor Effects Tool
Real estate video generation with parallel processing
- Phase 1a: Parallel uploads
- Phase 1b: Parallel outpaints (9:16 expansion)
- Phase 1c: Parallel transforms (Float, Reno, 3D text)
- Phase 2: Parallel video generation

UI: Icon grid for effects, per-shot redo controls
"""
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_client import (
    upload_image,
    outpaint_to_vertical,
    transform_image,
    generate_video,
    generate_video_kie,
    EFFECT_PROMPTS,
    EFFECT_VIDEO_MODEL,
    TRANSFORM_EFFECTS,
    NEEDS_OUTPAINT,
    TEXT_EFFECTS
)
import random
import zipfile
import io

st.set_page_config(
    page_title="BW Editor Effects Tool",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Effect grid buttons */
.effect-btn {
    height: 70px !important;
    font-size: 13px !important;
}
/* Results grid uniform sizing */
.result-video video {
    max-height: 350px;
    object-fit: contain;
}
/* Compact buttons */
div[data-testid="column"] .stButton button {
    padding: 0.25rem 0.5rem;
    font-size: 12px;
}
/* Shot Generator compact thumbnails */
.shot-thumb {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 4px;
}
.shot-status {
    font-size: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Effect configuration with icons
EFFECT_CONFIG = {
    "None": {"icon": "⬜", "desc": "Original only", "key": "none"},
    "VEO Cam": {"icon": "🎥", "desc": "Camera motion", "key": "veo_cam"},
    "Float": {"icon": "🎈", "desc": "Floating furniture", "key": "float"},
    "Day to Night": {"icon": "🌙", "desc": "Light transition", "key": "day_to_night"},
    "Staging Inside": {"icon": "🛋️", "desc": "Furniture reveal", "key": "staging_inside"},
    "Staging Outside": {"icon": "🌳", "desc": "Landscaping reveal", "key": "staging_outside"},
    "Reno": {"icon": "🔨", "desc": "Construction reveal", "key": "reno"},
    "3D Price": {"icon": "💰", "desc": "Price overlay", "key": "3d_price"},
    "3D City": {"icon": "🏙️", "desc": "City overlay", "key": "3d_city"},
    "3D Beds": {"icon": "🛏️", "desc": "Beds/baths overlay", "key": "3d_beds"},
}

EFFECTS = list(EFFECT_CONFIG.keys())

def get_effect_key(effect_name):
    return EFFECT_CONFIG.get(effect_name, {}).get("key", effect_name.lower().replace(" ", "_"))

# Session state initialization - Effects tab
if "videos" not in st.session_state:
    st.session_state.videos = {}
if "effect_choices" not in st.session_state:
    st.session_state.effect_choices = {}
if "jobs" not in st.session_state:
    st.session_state.jobs = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {}

# Session state initialization - Shot Generator tab
if "sg_results" not in st.session_state:
    st.session_state.sg_results = {}
if "sg_jobs" not in st.session_state:
    st.session_state.sg_jobs = {}
if "sg_prompts" not in st.session_state:
    st.session_state.sg_prompts = {}
if "sg_reversed" not in st.session_state:
    st.session_state.sg_reversed = set()
if "sg_edit_mode" not in st.session_state:
    st.session_state.sg_edit_mode = {}

# Title
st.title("🎬 BW Editor Effects Tool")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    duration = st.select_slider("Video Duration", options=["4s", "6s", "8s"], value="4s")
    
    st.divider()
    
    st.header("📝 3D Text Content")
    text_price = st.text_input("Price", "$1,250,000")
    text_city = st.text_input("City", "Austin, TX")
    text_beds = st.text_input("Beds/Baths", "4 Bed / 3 Bath")
    
    st.divider()
    
    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_effect_grid(image_idx: int, current_effect: str):
    """Render a grid of effect options with icons."""
    # 5 columns x 2 rows
    effects_list = EFFECTS
    
    for row in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            effect_idx = row * 5 + col_idx
            if effect_idx < len(effects_list):
                effect = effects_list[effect_idx]
                cfg = EFFECT_CONFIG[effect]
                is_selected = effect == current_effect
                
                with cols[col_idx]:
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(
                        f"{cfg['icon']}\n{effect}",
                        key=f"eff_{image_idx}_{effect_idx}",
                        type=btn_type,
                        use_container_width=True
                    ):
                        st.session_state.effect_choices[image_idx] = effect
                        st.rerun()


def redo_image_for_shot(idx: int, uploaded_files, text_price, text_city, text_beds):
    """Regenerate image (upload + outpaint + transform) for one shot."""
    job = st.session_state.jobs.get(idx)
    if not job:
        return
    
    file = uploaded_files[idx]
    image_bytes = file.read()
    file.seek(0)
    
    with st.spinner(f"Regenerating image for shot {idx + 1}..."):
        # Re-upload
        image_url = upload_image(image_bytes, file.name)
        job["image_url"] = image_url
        
        # Re-outpaint if needed
        if job["needs_outpaint"]:
            vertical_url = outpaint_to_vertical(image_url)
        else:
            vertical_url = image_url
        job["vertical_url"] = vertical_url
        
        # Re-transform if needed
        if job["needs_transform"]:
            end_frame = transform_image(vertical_url, job["effect_key"], job.get("text_content"), use_fal=True)
            job["end_frame"] = end_frame
        
        # Update frames based on effect type
        update_job_frames(job)
        
        # Update result frames (keep old video)
        if idx in st.session_state.videos:
            st.session_state.videos[idx]["start_frame"] = job["start_frame"]
            st.session_state.videos[idx]["end_frame"] = job["end_frame"]
            st.session_state.videos[idx]["image_regenerated"] = True
        
        st.session_state.jobs[idx] = job


def redo_video_for_shot(idx: int, duration: str):
    """Regenerate video from current frames."""
    job = st.session_state.jobs.get(idx)
    if not job:
        return
    
    with st.spinner(f"Regenerating video for shot {idx + 1}..."):
        video_url = generate_video(
            job["start_frame"],
            job.get("end_frame", job["start_frame"]),
            job["prompt"],
            duration,
            model=job["video_model"]
        )
        
        st.session_state.videos[idx]["video_url"] = video_url
        st.session_state.videos[idx]["image_regenerated"] = False


def update_job_frames(job):
    """Set start/end frames based on effect type."""
    if job["is_text_effect"]:
        job["start_frame"] = job.get("end_frame", job["vertical_url"])
        job["end_frame"] = job.get("end_frame", job["vertical_url"])
    elif job["effect_key"] == "staging_inside":
        job["start_frame"] = job.get("end_frame", job["vertical_url"])
        job["end_frame"] = job["vertical_url"]
    elif job["effect_key"] == "staging_outside":
        job["start_frame"] = job.get("end_frame", job["vertical_url"])
        job["end_frame"] = job["vertical_url"]
    elif job["effect_key"] == "reno":
        job["start_frame"] = job["vertical_url"]
        # end_frame already set
    else:
        job["start_frame"] = job["vertical_url"]
        job["end_frame"] = job["vertical_url"]


def render_result_card(idx: int, result: dict, uploaded_files, duration, text_price, text_city, text_beds):
    """Render a single result card with video and controls."""
    with st.container(border=True):
        st.markdown(f"**Shot {idx + 1}: {result['effect']}**")
        
        if result["status"] == "success":
            # Main video
            st.video(result["video_url"])
            
            # Frame preview (collapsed)
            with st.expander("📷 View Frames", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Start Frame")
                    st.image(result.get("start_frame", ""), use_container_width=True)
                with c2:
                    st.caption("End Frame")
                    if result.get("end_frame") != result.get("start_frame"):
                        st.image(result.get("end_frame", ""), use_container_width=True)
                    else:
                        st.info("Same as start")
            
            # Check if in prompt edit mode
            if st.session_state.edit_mode.get(idx) == "prompt":
                job = st.session_state.jobs.get(idx, {})
                current_prompt = job.get("prompt", "")
                new_prompt = st.text_area(
                    "Edit video prompt",
                    value=current_prompt,
                    key=f"prompt_edit_{idx}",
                    height=80
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save", key=f"save_prompt_{idx}", use_container_width=True):
                        st.session_state.jobs[idx]["prompt"] = new_prompt
                        st.session_state.edit_mode[idx] = None
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel", key=f"cancel_prompt_{idx}", use_container_width=True):
                        st.session_state.edit_mode[idx] = None
                        st.rerun()
            else:
                # Action buttons
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    if st.button("🔄 Redo Img", key=f"redo_img_{idx}", use_container_width=True):
                        redo_image_for_shot(idx, uploaded_files, text_price, text_city, text_beds)
                        st.rerun()
                
                with c2:
                    if st.button("✏️ Prompt", key=f"edit_prompt_{idx}", use_container_width=True):
                        st.session_state.edit_mode[idx] = "prompt"
                        st.rerun()
                
                with c3:
                    regen_label = "🎬 Redo Vid" if not result.get("image_regenerated") else "🎬 Gen Vid!"
                    if st.button(regen_label, key=f"redo_vid_{idx}", use_container_width=True):
                        redo_video_for_shot(idx, duration)
                        st.rerun()
                
                with c4:
                    st.download_button(
                        "⬇️",
                        data=requests.get(result["video_url"]).content,
                        file_name=f"video_{idx + 1}.mp4",
                        mime="video/mp4",
                        key=f"dl_{idx}",
                        use_container_width=True
                    )
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")


# Main content - Tabs
tab_effects, tab_shotgen = st.tabs(["🎨 Effects", "📹 Shot Generator"])

# ==================== EFFECTS TAB ====================
with tab_effects:
    st.subheader("📤 Upload Images & Assign Effects")

    uploaded_files = st.file_uploader(
        "Upload listing images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="effects_uploader"
    )

    if uploaded_files:
        num_files = len(uploaded_files)

        # Initialize effect choices
        for i in range(num_files):
            if i not in st.session_state.effect_choices:
                st.session_state.effect_choices[i] = "VEO Cam"

        # Display images with effect grids
        for img_idx in range(num_files):
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(uploaded_files[img_idx], use_container_width=True)
                    st.caption(f"Image {img_idx + 1}")

                with col2:
                    current_effect = st.session_state.effect_choices.get(img_idx, "VEO Cam")
                    st.markdown(f"**Selected: {current_effect}**")
                    render_effect_grid(img_idx, current_effect)

        st.divider()

        # Generate button
        if st.button("🚀 Generate All Videos", type="primary", use_container_width=True):
            st.session_state.videos = {}
            st.session_state.jobs = {}
            progress = st.progress(0)
            status = st.empty()

            # Prepare all jobs
            jobs = []
            for i, file in enumerate(uploaded_files):
                effect_name = st.session_state.effect_choices[i]
                effect_key = get_effect_key(effect_name)
                image_bytes = file.read()
                file.seek(0)

                # Skip "None" effect
                if effect_key == "none":
                    st.session_state.videos[i] = {
                        "idx": i,
                        "status": "skipped",
                        "effect": effect_name
                    }
                    continue

                # Get text content for 3D text effects
                text_content = None
                if effect_key == "3d_price":
                    text_content = text_price
                elif effect_key == "3d_city":
                    text_content = text_city
                elif effect_key == "3d_beds":
                    text_content = text_beds

                video_prompt = EFFECT_PROMPTS.get(effect_key, EFFECT_PROMPTS["veo_cam"])

                job = {
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
                    "prompt": video_prompt
                }
                jobs.append(job)
                st.session_state.jobs[i] = job

            if not jobs:
                st.warning("All images set to 'None' - nothing to generate")
                st.stop()

            total_steps = len(jobs) * 3
            completed_steps = 0

            # === PHASE 1a: Parallel uploads ===
            status.text(f"[Phase 1a] Uploading {len(jobs)} images...")

            def upload_job(job):
                url = upload_image(job["image_bytes"], job["filename"])
                return {"idx": job["idx"], "url": url}

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(upload_job, j): j for j in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    idx = result["idx"]
                    for j in jobs:
                        if j["idx"] == idx:
                            j["image_url"] = result["url"]
                            st.session_state.jobs[idx]["image_url"] = result["url"]
                    completed_steps += 1
                    progress.progress(completed_steps / total_steps)

            # === PHASE 1b: Parallel outpaints ===
            outpaint_jobs = [j for j in jobs if j["needs_outpaint"]]

            if outpaint_jobs:
                status.text(f"[Phase 1b] Expanding {len(outpaint_jobs)} images to 9:16...")

                def outpaint_job(job):
                    vertical = outpaint_to_vertical(job["image_url"])
                    return {"idx": job["idx"], "vertical": vertical}

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(outpaint_job, j): j for j in outpaint_jobs}
                    for future in as_completed(futures):
                        result = future.result()
                        idx = result["idx"]
                        for j in jobs:
                            if j["idx"] == idx:
                                j["vertical_url"] = result["vertical"]
                                st.session_state.jobs[idx]["vertical_url"] = result["vertical"]
                        completed_steps += 1
                        progress.progress(completed_steps / total_steps)

            for job in jobs:
                if not job["needs_outpaint"]:
                    job["vertical_url"] = job["image_url"]
                    st.session_state.jobs[job["idx"]]["vertical_url"] = job["image_url"]
                    completed_steps += 1
                    progress.progress(completed_steps / total_steps)

            # === PHASE 1c: Parallel transforms ===
            transform_jobs = [j for j in jobs if j["needs_transform"]]

            if transform_jobs:
                status.text(f"[Phase 1c] Transforming {len(transform_jobs)} images...")

                def transform_job(job):
                    transformed = transform_image(job["vertical_url"], job["effect_key"], job.get("text_content"), use_fal=True)
                    return {"idx": job["idx"], "transformed": transformed}

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(transform_job, j): j for j in transform_jobs}
                    for future in as_completed(futures):
                        result = future.result()
                        idx = result["idx"]
                        for j in jobs:
                            if j["idx"] == idx:
                                j["end_frame"] = result["transformed"]
                                st.session_state.jobs[idx]["end_frame"] = result["transformed"]

            # Set frames for all jobs
            for job in jobs:
                update_job_frames(job)
                st.session_state.jobs[job["idx"]]["start_frame"] = job["start_frame"]
                st.session_state.jobs[job["idx"]]["end_frame"] = job.get("end_frame", job["start_frame"])

            # === PHASE 2: Parallel video generation ===
            status.text(f"[Phase 2] Generating {len(jobs)} videos...")

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
                        "end_frame": job.get("end_frame", job["start_frame"]),
                        "image_regenerated": False
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
            st.subheader("📹 Generated Videos")

            # Results in 2-column grid
            results_to_show = [(i, st.session_state.videos[i]) for i in range(num_files) if i in st.session_state.videos and st.session_state.videos[i].get("status") != "skipped"]

            for row_start in range(0, len(results_to_show), 2):
                cols = st.columns(2)
                for col_idx in range(2):
                    result_idx = row_start + col_idx
                    if result_idx < len(results_to_show):
                        idx, result = results_to_show[result_idx]
                        with cols[col_idx]:
                            render_result_card(idx, result, uploaded_files, duration, text_price, text_city, text_beds)

    else:
        st.info("👆 Upload listing images to get started")

        # Effect reference
        st.subheader("Available Effects")
        cols = st.columns(5)
        for i, (name, cfg) in enumerate(EFFECT_CONFIG.items()):
            with cols[i % 5]:
                st.markdown(f"**{cfg['icon']} {name}**")
                st.caption(cfg['desc'])

# ==================== SHOT GENERATOR TAB ====================
with tab_shotgen:
    st.subheader("📹 Shot Generator")
    st.caption("Upload 20-30 images, apply a global prompt, generate videos with Kling 2.1")

    # Upload section
    sg_files = st.file_uploader(
        "Upload images (20-30 recommended)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="shotgen_uploader"
    )

    if sg_files:
        num_sg = len(sg_files)

        # Compact 8-column thumbnail grid
        st.markdown("**Uploaded Images**")
        thumb_cols = st.columns(8)
        for i, f in enumerate(sg_files):
            with thumb_cols[i % 8]:
                st.image(f, use_container_width=True)
                st.caption(f"{i+1}")

        st.divider()

        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 1, 1])

        with ctrl_col1:
            sg_prompt = st.text_input(
                "Global Prompt",
                value="Slow, smooth camera move forward, professional real estate video.",
                key="sg_global_prompt"
            )

        with ctrl_col2:
            sg_duration = st.selectbox("Duration", ["5s", "10s"], key="sg_duration")

        with ctrl_col3:
            sg_reverse = st.checkbox("Reverse middle (40%)", key="sg_reverse_middle")

        # Generate button
        if st.button("🚀 Generate All Videos", type="primary", use_container_width=True, key="sg_generate"):
            st.session_state.sg_results = {}
            st.session_state.sg_jobs = {}
            st.session_state.sg_reversed = set()

            progress = st.progress(0)
            status = st.empty()

            # Prepare jobs
            jobs = []
            dur = sg_duration.replace("s", "")

            for i, f in enumerate(sg_files):
                image_bytes = f.read()
                f.seek(0)

                # Determine if this shot should be reversed (40% of middle shots)
                is_reversed = False
                if sg_reverse and 0 < i < num_sg - 1:  # Not first or last
                    if random.random() < 0.4:
                        is_reversed = True
                        st.session_state.sg_reversed.add(i)

                # Upload image to get URL
                image_url = upload_image(image_bytes, f.name)

                job = {
                    "idx": i,
                    "image_url": image_url,
                    "image_bytes": image_bytes,
                    "filename": f.name,
                    "prompt": sg_prompt,
                    "duration": dur,
                    "is_reversed": is_reversed
                }
                jobs.append(job)
                st.session_state.sg_jobs[i] = job

            status.text(f"Generating {len(jobs)} videos with VEO 3.1 Fast (KIE API)...")

            # Parallel video generation
            def gen_sg_video(job):
                try:
                    video_url = generate_video_kie(
                        job["image_url"],
                        job["prompt"],
                        "9:16"  # Portrait format
                    )
                    return {
                        "idx": job["idx"],
                        "status": "success",
                        "video_url": video_url,
                        "image_url": job["image_url"],
                        "is_reversed": job["is_reversed"]
                    }
                except Exception as e:
                    return {
                        "idx": job["idx"],
                        "status": "error",
                        "error": str(e)
                    }

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(gen_sg_video, j): j for j in jobs}
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    st.session_state.sg_results[result["idx"]] = result
                    completed += 1
                    progress.progress(completed / len(jobs))

            status.text("✅ Complete!")
            st.rerun()

        # Results section
        if st.session_state.sg_results:
            st.divider()
            st.subheader("📹 Generated Videos")

            # Download All ZIP button
            success_results = [r for r in st.session_state.sg_results.values() if r.get("status") == "success"]
            if success_results:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for r in success_results:
                        try:
                            video_data = requests.get(r["video_url"]).content
                            zf.writestr(f"shot_{r['idx']+1}.mp4", video_data)
                        except:
                            pass
                zip_buffer.seek(0)

                st.download_button(
                    "⬇️ Download All (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="shot_generator_videos.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            # Results grid - 4 columns
            results_list = [(i, st.session_state.sg_results[i]) for i in range(num_sg) if i in st.session_state.sg_results]

            for row_start in range(0, len(results_list), 4):
                cols = st.columns(4)
                for col_idx in range(4):
                    result_idx = row_start + col_idx
                    if result_idx < len(results_list):
                        idx, result = results_list[result_idx]
                        with cols[col_idx]:
                            with st.container(border=True):
                                rev_tag = " 🔄" if result.get("is_reversed") else ""
                                st.markdown(f"**Shot {idx+1}{rev_tag}**")

                                if result["status"] == "success":
                                    st.video(result["video_url"])

                                    # Redo controls
                                    if st.session_state.sg_edit_mode.get(idx) == "prompt":
                                        job = st.session_state.sg_jobs.get(idx, {})
                                        new_prompt = st.text_area(
                                            "Prompt",
                                            value=job.get("prompt", sg_prompt),
                                            key=f"sg_prompt_{idx}",
                                            height=60
                                        )
                                        c1, c2 = st.columns(2)
                                        with c1:
                                            if st.button("💾", key=f"sg_save_{idx}", use_container_width=True):
                                                st.session_state.sg_jobs[idx]["prompt"] = new_prompt
                                                st.session_state.sg_edit_mode[idx] = None
                                                st.rerun()
                                        with c2:
                                            if st.button("❌", key=f"sg_cancel_{idx}", use_container_width=True):
                                                st.session_state.sg_edit_mode[idx] = None
                                                st.rerun()
                                    else:
                                        c1, c2, c3 = st.columns(3)
                                        with c1:
                                            if st.button("✏️", key=f"sg_edit_{idx}", use_container_width=True):
                                                st.session_state.sg_edit_mode[idx] = "prompt"
                                                st.rerun()
                                        with c2:
                                            if st.button("🔄", key=f"sg_redo_{idx}", use_container_width=True):
                                                job = st.session_state.sg_jobs.get(idx)
                                                if job:
                                                    with st.spinner(f"Regenerating shot {idx+1}..."):
                                                        try:
                                                            video_url = generate_video_kie(
                                                                job["image_url"],
                                                                job["prompt"],
                                                                "9:16"  # Portrait format
                                                            )
                                                            st.session_state.sg_results[idx] = {
                                                                "idx": idx,
                                                                "status": "success",
                                                                "video_url": video_url,
                                                                "image_url": job["image_url"],
                                                                "is_reversed": job.get("is_reversed", False)
                                                            }
                                                        except Exception as e:
                                                            st.session_state.sg_results[idx] = {
                                                                "idx": idx,
                                                                "status": "error",
                                                                "error": str(e)
                                                            }
                                                        st.rerun()
                                        with c3:
                                            st.download_button(
                                                "⬇️",
                                                data=requests.get(result["video_url"]).content,
                                                file_name=f"shot_{idx+1}.mp4",
                                                mime="video/mp4",
                                                key=f"sg_dl_{idx}",
                                                use_container_width=True
                                            )
                                else:
                                    st.error(f"Error: {result.get('error', 'Unknown')}"[:50])
    else:
        st.info("👆 Upload 20-30 images to generate a video sequence")
