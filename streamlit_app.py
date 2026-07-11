import os
import tempfile
import time
import json
import streamlit as st
from PIL import Image

# Prevent OpenAI missing credentials crash on startup/import before importing pipeline
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "sk_placeholder_key_to_prevent_import_error"

# Add root directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Sentinel-V pipeline components safely
from src.config import VISION_MODEL, TEXT_MODEL, SPEECH_MODEL
from src.pipeline.extract import extract_dynamic_keyframes
from src.pipeline.transcribe import extract_and_transcribe_audio
from src.pipeline.caption import generate_styled_captions
from src.agent import TokenBucketScheduler, download_video

# Page Configuration
st.set_page_config(
    page_title="Sentinel-V: AI Video Captioning Hub",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0B0F19;
    }
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: -webkit-linear-gradient(left, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #94A3B8;
        margin-bottom: 30px;
    }
    .caption-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #161F38;
        border-left: 5px solid #38BDF8;
    }
    .formal-card { border-left-color: #38BDF8; }
    .sarcastic-card { border-left-color: #F87171; }
    .tech-card { border-left-color: #34D399; }
    .non-tech-card { border-left-color: #FBBF24; }
    
    .card-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .card-content {
        font-size: 14px;
        line-height: 1.5;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-title">🎬 SENTINEL-V CAPTIONING HUB</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High-Performance, Multi-Style Video Captioning powered by Llama 4 Scout, Llama 3.3, and Whisper Large V3.</div>', unsafe_allow_html=True)

# Sidebar Configurations
st.sidebar.image("https://img.icons8.com/nolan/128/video-playlist.png", width=80)
st.sidebar.title("Configuration")

# 1. API Authentication
current_env_key = os.environ.get("GROQ_API_KEY", "")
display_key = "" if current_env_key == "sk_placeholder_key_to_prevent_import_error" else current_env_key

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=display_key,
    type="password",
    help="Enter your Groq API key. If already set in environment variables, it will load automatically."
)

if groq_api_key and groq_api_key != "sk_placeholder_key_to_prevent_import_error":
    os.environ["GROQ_API_KEY"] = groq_api_key
    # Hot-patch the settings and clients in other modules using sys.modules to prevent namespace conflicts
    import sys
    config_mod = sys.modules.get("src.config")
    transcribe_mod = sys.modules.get("src.pipeline.transcribe")
    caption_mod = sys.modules.get("src.pipeline.caption")
    
    if config_mod:
        config_mod.GROQ_API_KEY = groq_api_key
    if transcribe_mod:
        transcribe_mod.GROQ_API_KEY = groq_api_key
    if caption_mod and hasattr(caption_mod, "groq_client"):
        caption_mod.groq_client.api_key = groq_api_key

# 2. Model information (Read-only representation)
st.sidebar.subheader("AI Models")
st.sidebar.info(f"👁️ Vision: {VISION_MODEL}\n\n📝 Text: {TEXT_MODEL}\n\n🗣️ Audio: {SPEECH_MODEL}")

# 3. Parameters
st.sidebar.subheader("Parameters")
target_frame_count = st.sidebar.slider("Keyframes to Extract", min_value=5, max_value=15, value=10)
tpm_limit_vision = st.sidebar.number_input("Vision TPM Limit", value=28000)
tpm_limit_text = st.sidebar.number_input("Text TPM Limit", value=11000)

# Initialize Scheduler
scheduler = TokenBucketScheduler(
    tpm_limit_vision=tpm_limit_vision,
    tpm_limit_text=tpm_limit_text
)

# Input Section: Choose URL or Upload
st.subheader("1. Ingest Video Source")
input_method = st.radio("Choose Input Method", ["Direct Video URL", "Upload Video File"], horizontal=True)

video_path = None
work_dir = None

# Create working directory
if "work_dir_path" not in st.session_state:
    st.session_state.work_dir_path = tempfile.mkdtemp(prefix="streamlit_sentinel_")

work_dir = st.session_state.work_dir_path
os.makedirs(work_dir, exist_ok=True)

if input_method == "Direct Video URL":
    video_url = st.text_input(
        "Direct MP4 Link",
        value="https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
        placeholder="https://example.com/video.mp4"
    )
    if video_url:
        st.video(video_url)
        video_path = os.path.join(work_dir, "downloaded_video.mp4")
        
        # Download logic when requested
        download_btn = st.button("Load and Pre-Process URL")
        if download_btn:
            with st.spinner("Downloading video stream..."):
                if download_video(video_url, video_path):
                    st.success("Video downloaded successfully!")
                else:
                    st.error("Failed to download video from URL.")
                    video_path = None
else:
    uploaded_file = st.file_uploader("Upload Video File (.mp4, .mov)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        video_path = os.path.join(work_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)
        st.success("File uploaded successfully!")

# Selection of Styles
st.subheader("2. Select Caption Styles")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    style_formal = st.checkbox("Formal Style", value=True)
with col_s2:
    style_sarcastic = st.checkbox("Sarcastic Style", value=True)
with col_s3:
    style_tech = st.checkbox("Humorous Tech", value=True)
with col_s4:
    style_non_tech = st.checkbox("Humorous Non-Tech", value=True)

requested_styles = []
if style_formal: requested_styles.append("formal")
if style_sarcastic: requested_styles.append("sarcastic")
if style_tech: requested_styles.append("humorous_tech")
if style_non_tech: requested_styles.append("humorous_non_tech")

# Process Button
st.subheader("3. Execution")
if st.button("🚀 Run Captioning Pipeline", disabled=(video_path is None or not requested_styles)):
    if not groq_api_key:
        st.error("Please enter a Groq API Key in the sidebar to authenticate calls.")
    else:
        # Progress Tracking via Streamlit Status UI
        with st.status("Processing video captioning agent...") as status:
            
            # Step 1: Feature Extraction
            status.update(label="Extracting keyframes using FFMPEG scene cut detection...", state="running")
            frames_dir = os.path.join(work_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            keyframes = extract_dynamic_keyframes(video_path, frames_dir, target_count=target_frame_count)
            st.write(f"✓ Extracted {len(keyframes)} keyframes.")
            
            # Show frames in Expander
            if keyframes:
                with st.expander("Show Extracted Keyframes"):
                    cols = st.columns(min(len(keyframes), 5))
                    for idx, kf_path in enumerate(keyframes):
                        col_idx = idx % 5
                        img = Image.open(kf_path)
                        cols[col_idx].image(img, caption=f"Frame {idx+1}", use_column_width=True)
            else:
                st.warning("⚠️ No keyframes were extracted. Visual analysis might fail.")
            
            # Step 2: Speech Transcription
            status.update(label="Extracting audio track and transcribing via Groq Whisper...", state="running")
            audio_transcript = extract_and_transcribe_audio(video_path, work_dir)
            if audio_transcript:
                st.write("✓ Audio Transcription complete.")
                st.info(f"**Speech Detected:** '{audio_transcript}'")
            else:
                st.write("✓ Audio Transcription completed (no speech/silence detected).")
            
            # Step 3: LLM Generation
            status.update(label="Synthesizing visual narrative and generating styled captions...", state="running")
            start_gen_time = time.time()
            raw_captions = generate_styled_captions(keyframes, audio_transcript, scheduler)
            final_captions = {style: raw_captions.get(style, "Caption unavailable.") for style in requested_styles}
            st.write(f"✓ Gen completed in {time.time() - start_gen_time:.2f}s.")
            
            status.update(label="Pipeline Executed Successfully!", state="complete")

        # Results Display Area
        st.subheader("🎬 Generated Styled Captions")
        
        c_cols = st.columns(len(requested_styles))
        for idx, style in enumerate(requested_styles):
            col = c_cols[idx]
            caption = final_captions.get(style)
            
            # Apply styling class depending on the style
            card_class = "caption-card "
            accent_title = ""
            if style == "formal":
                card_class += "formal-card"
                accent_title = "🤖 Formal (HAL-9000)"
            elif style == "sarcastic":
                card_class += "sarcastic-card"
                accent_title = "🙄 Sarcastic"
            elif style == "humorous_tech":
                card_class += "tech-card"
                accent_title = "💻 Humorous (Tech)"
            elif style == "humorous_non_tech":
                card_class += "non-tech-card"
                accent_title = "👨‍🦳 Humorous (Non-Tech)"
                
            col.markdown(f"""
            <div class="{card_class}">
                <div class="card-title">{accent_title}</div>
                <div class="card-content">"{caption}"</div>
            </div>
            """, unsafe_allow_html=True)
            
        # JSON Output Expander
        with st.expander("View Raw JSON Output"):
            results_json = [{
                "task_id": "streamlit_demo",
                "captions": final_captions
            }]
            st.code(json.dumps(results_json, indent=2), language="json")
            st.download_button(
                label="📥 Download results.json",
                data=json.dumps(results_json, indent=2),
                file_name="results.json",
                mime="application/json"
            )
else:
    if video_path is None:
        st.warning("Please upload a video or click 'Load and Pre-Process URL' to begin.")
