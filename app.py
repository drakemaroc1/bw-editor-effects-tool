"""
Editor Effects Tool — Streamlit MVP v2
Cleaner UI with auto-assignment and smaller thumbnails
"""

import streamlit as st
from pathlib import Path
from api_client import (
    generate_nano_banana_image,
    generate_effect_frames,
    get_camera_movement_prompt
)

# Page config
st.set_page_config(
    page_title="BW Editor Effects",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for smaller thumbnails
st.markdown("""
<style>
    .small-thumb {
        max-height: 120px;
        object-fit: cover;
    }
    .stSelectbox > div > div {
        font-size: 12px;
    }
    div[data-testid="column"] {
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 Editor Effects Tool")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Listing Info")
    price = st.text_input("Price", "$749,000")
    city = st.text_input("City", "Austin, TX")
    beds_baths = st.text_input("Beds/Baths", "4 Bed • 3 Bath")
    property_desc = st.text_input("Description", "Modern home with pool")
    
    st.markdown("---")
    
    # Quick assign buttons
    st.header("⚡ Quick Assign")
    if st.button("🌅 All Day→Night", use_container_width=True):
        st.session_state.bulk_assign = "Lighting Transform (Day→Night)"
    if st.button("📹 All Orbit", use_container_width=True):
        st.session_state.bulk_assign = "Orbit"
    if st.button("🎯 Auto Mix", use_container_width=True):
        st.session_state.bulk_assign = "auto_mix"

# Effect options (shorter labels)
EFFECTS = {
    "Day→Night": "Lighting Transform (Day→Night)",
    "Staging": "Virtual Staging (Empty→Furnished)",
    "Float": "Furniture Float", 
    "Reno": "Construction → Finished",
    "Punch-in": "Close-up Punch-in",
    "Orbit": "Orbit",
    "Dolly": "Dolly In",
    "Crane": "Crane Up",
    "Push": "Push",
    "Pull": "Pull/Reveal",
    "3D Price": "Push + 3D Text (Price)",
    "3D City": "Push + 3D Text (City)",
    "3D Beds": "Push + 3D Text (Beds/Baths)",
}

EFFECT_LABELS = list(EFFECTS.keys())

# Auto-mix assignment pattern
AUTO_MIX = ["Day→Night", "Orbit", "3D Price", "Dolly", "Staging", "Crane", "3D City", "Push", "Float", "Pull"]

# Session state
if 'assignments' not in st.session_state:
    st.session_state.assignments = {}
if 'generated_results' not in st.session_state:
    st.session_state.generated_results = {}
if 'bulk_assign' not in st.session_state:
    st.session_state.bulk_assign = None

# --- UPLOAD ---
uploaded_files = st.file_uploader(
    "📁 Drop listing images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    num_files = len(uploaded_files)
    
    # Handle bulk assignment
    if st.session_state.bulk_assign:
        if st.session_state.bulk_assign == "auto_mix":
            for i in range(num_files):
                st.session_state.assignments[i] = AUTO_MIX[i % len(AUTO_MIX)]
        else:
            for i in range(num_files):
                # Find short label for the full effect name
                for short, full in EFFECTS.items():
                    if full == st.session_state.bulk_assign:
                        st.session_state.assignments[i] = short
                        break
        st.session_state.bulk_assign = None
        st.rerun()
    
    # Initialize assignments if needed
    for i in range(num_files):
        if i not in st.session_state.assignments:
            st.session_state.assignments[i] = AUTO_MIX[i % len(AUTO_MIX)]
    
    st.markdown("### 🖼️ Assign Effects (click to change)")
    
    # Grid with 5 columns for smaller thumbnails
    cols = st.columns(5)
    
    for i, file in enumerate(uploaded_files):
        with cols[i % 5]:
            st.image(file, use_container_width=True)
            current = st.session_state.assignments.get(i, "Day→Night")
            new_val = st.selectbox(
                f"#{i+1}",
                EFFECT_LABELS,
                index=EFFECT_LABELS.index(current) if current in EFFECT_LABELS else 0,
                key=f"sel_{i}",
                label_visibility="collapsed"
            )
            st.session_state.assignments[i] = new_val
    
    st.markdown("---")
    
    # Summary
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**{num_files} images** ready to process")
    with col2:
        effect_count = sum(1 for a in st.session_state.assignments.values() if a in ["Day→Night", "Staging", "Float", "Reno", "Punch-in"])
        st.markdown(f"🎨 {effect_count} effects")
    with col3:
        camera_count = num_files - effect_count
        st.markdown(f"📹 {camera_count} camera")
    
    # Generate button
    if st.button("🚀 GENERATE ALL", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        results = {}
        
        for i, file in enumerate(uploaded_files):
            short_label = st.session_state.assignments.get(i, "Day→Night")
            effect_type = EFFECTS.get(short_label, short_label)
            
            status.text(f"Processing {i+1}/{num_files}: {short_label}...")
            
            # Get text content for 3D effects
            text_content = None
            if "Price" in effect_type:
                text_content = price
            elif "City" in effect_type:
                text_content = city
            elif "Beds" in effect_type:
                text_content = beds_baths
            
            try:
                if short_label in ["Orbit", "Dolly", "Crane", "Push", "Pull"]:
                    # Camera only
                    prompt = get_camera_movement_prompt(property_desc, effect_type)
                    results[i] = {
                        'type': 'camera',
                        'label': short_label,
                        'prompt': prompt,
                        'status': '✅'
                    }
                else:
                    # Effect with frames
                    frame_result = generate_effect_frames(property_desc, effect_type, text_content)
                    results[i] = {
                        'type': 'effect',
                        'label': short_label,
                        **frame_result,
                        'status': '✅'
                    }
            except Exception as e:
                results[i] = {
                    'type': 'error',
                    'label': short_label,
                    'error': str(e),
                    'status': '❌'
                }
            
            progress.progress((i + 1) / num_files)
        
        st.session_state.generated_results = results
        status.text("✅ Done!")
    
    # Results
    if st.session_state.generated_results:
        st.markdown("---")
        st.markdown("### 📤 Results")
        
        results = st.session_state.generated_results
        
        for i, (idx, result) in enumerate(results.items()):
            with st.expander(f"{result['status']} Shot {idx+1}: {result['label']}", expanded=False):
                if result.get('error'):
                    st.error(result['error'])
                elif result['type'] == 'camera':
                    st.code(result['prompt'], language=None)
                else:
                    col1, col2 = st.columns(2)
                    if result.get('start_frame'):
                        with col1:
                            st.image(result['start_frame'], caption="Start")
                    if result.get('end_frame'):
                        with col2:
                            st.image(result['end_frame'], caption="End")
                    elif result.get('start_frame'):
                        st.info("Single frame effect")

else:
    st.info("👆 Upload images to get started")
    
    # Show effect options
    with st.expander("Available effects"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎨 Effects (frame generation)**")
            st.markdown("- Day→Night\n- Staging\n- Float\n- Reno\n- Punch-in")
        with col2:
            st.markdown("**📹 Camera (VEO prompt)**")
            st.markdown("- Orbit\n- Dolly\n- Crane\n- Push\n- Pull")
