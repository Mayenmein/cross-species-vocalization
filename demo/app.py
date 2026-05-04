# demo/app.py
"""
Streamlit demo application for Cross-Species Vocalization Classifier.

Run with:
    streamlit run demo/app.py

Features:
- Upload audio files for classification
- Record live audio from microphone
- View classification results with confidence scores
- See attention visualization (which parts of audio mattered most)
- Batch process multiple files
- Download results as CSV
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import tempfile
import time
import soundfile as sf
from io import BytesIO
import base64

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.classifier import AudioClassifier
from inference.live_mic import LiveMicrophoneClassifier
from inference.batch_process import BatchProcessor
from models.config import SOUND_CLASSES, DOMAIN_MAP, CLASS_TO_DOMAIN


# ─── Page Configuration ───
st.set_page_config(
    page_title="Cross-Species Vocalization Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        transition: width 0.5s ease;
    }
    .domain-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .domain-animal { background: #2ecc71; color: white; }
    .domain-human_nonverbal { background: #3498db; color: white; }
    .domain-machinery { background: #e74c3c; color: white; }
    .audio-player {
        width: 100%;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ───
if "classifier" not in st.session_state:
    @st.cache_resource
    def load_classifier():
        """Load model once and cache it."""
        model_path = "checkpoints/final_model.pt"
        if not Path(model_path).exists():
            st.warning("No trained model found. Using untrained model for demo.")
        return AudioClassifier(model_path=model_path)
    
    st.session_state.classifier = load_classifier()

if "history" not in st.session_state:
    st.session_state.history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ─── Helper Functions ───
def get_domain_emoji(domain: str) -> str:
    """Get emoji for domain."""
    emojis = {
        "animal": "🐾",
        "human_nonverbal": "👤",
        "machinery": "⚙️",
        "unknown": "❓",
    }
    return emojis.get(domain, "❓")


def create_confidence_chart(top5: list) -> go.Figure:
    """Create horizontal bar chart of top-5 predictions."""
    names = [t["class_name"].replace("_", " ").title() for t in top5]
    confidences = [t["confidence"] for t in top5]
    domains = [t["domain"] for t in top5]
    
    colors = {
        "animal": "#2ecc71",
        "human_nonverbal": "#3498db",
        "machinery": "#e74c3c",
    }
    bar_colors = [colors.get(d, "#95a5a6") for d in domains]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences[::-1],
            y=names[::-1],
            orientation="h",
            marker_color=bar_colors[::-1],
            text=[f"{c:.1%}" for c in confidences[::-1]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    ])
    
    fig.update_layout(
        xaxis_title="Confidence",
        xaxis_range=[0, 1],
        xaxis_tickformat=".0%",
        showlegend=False,
        height=250,
        margin=dict(l=0, r=50, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig


def create_domain_pie(domain_distribution: dict) -> go.Figure:
    """Create pie chart of domain distribution."""
    if not domain_distribution:
        return None
    
    labels = [d.replace("_", " ").title() for d in domain_distribution.keys()]
    values = list(domain_distribution.values())
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors[:len(labels)],
            hole=0.4,
            textinfo="label+percent",
        )
    ])
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig


def audio_to_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convert audio data to base64 for HTML audio player."""
    buffer = BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{b64}"


# ─── Main UI ───
st.markdown('<h1 class="main-header">🎵 Cross-Species Vocalization Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Fine-tuned Whisper for animal calls · human sounds · machinery diagnostics</p>', unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Domain selection
    domain = st.selectbox(
        "Classification Domain",
        options=["auto", "animal", "human_nonverbal", "machinery"],
        format_func=lambda x: {
            "auto": "🤖 Auto-detect",
            "animal": "🐾 Animal Sounds",
            "human_nonverbal": "👤 Human Sounds",
            "machinery": "⚙️ Machinery Sounds",
        }.get(x, x),
        help="Select domain or let the model auto-detect"
    )
    
    st.divider()
    
    # Navigation
    st.header("📋 Mode")
    mode = st.radio(
        "Select Mode",
        options=["Upload File", "Record Live", "Batch Process", "History"],
        format_func=lambda x: {
            "Upload File": "📁 Upload File",
            "Record Live": "🎤 Record Live",
            "Batch Process": "📦 Batch Process",
            "History": "📜 History",
        }.get(x, x),
    )
    
    st.divider()
    
    # Info
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Cross-Species Vocalization Classifier**
        
        Fine-tuned OpenAI Whisper model that classifies sounds across three domains:
        
        - 🐾 **Animal**: Dog barks, cat meows, whale calls, bird songs
        - 👤 **Human**: Coughs, cries, laughter, gasps
        - ⚙️ **Machinery**: Engines, alarms, grinding, hissing
        
        Built with PyTorch and Streamlit.
        """)
    
    with st.expander("📊 Model Info"):
        st.markdown(f"""
        - **Architecture**: Whisper tiny + LSTM + Attention
        - **Parameters**: 39M (Whisper) + 2M (new layers)
        - **Input**: 10-second audio, 16kHz mono
        - **Classes**: 20 sound categories
        - **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}
        """)


# ─── Mode 1: Upload File ───
if mode == "Upload File":
    st.header("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Upload a sound file to classify"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        
        # Display audio player
        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix[1:]}")
        
        # Classify button
        if st.button("🔍 Classify Sound", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                start_time = time.time()
                result = st.session_state.classifier.classify(
                    temp_path,
                    domain=domain if domain != "auto" else "animal",
                )
                inference_time = time.time() - start_time
            
            # Add to history
            st.session_state.history.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "filename": uploaded_file.name,
                "result": result,
            })
            
            # Display results
            pred = result["prediction"]
            
            # Main prediction card
            emoji = get_domain_emoji(pred["domain"])
            st.markdown(f"""
            <div class="prediction-card">
                <h2>{emoji} {pred['class_name'].replace('_', ' ').title()}</h2>
                <h1 style="font-size: 3rem;">{pred['confidence']:.1%}</h1>
                <p style="opacity: 0.8;">confidence</p>
                <span class="domain-badge domain-{pred['domain']}">{pred['domain'].replace('_', ' ')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div style="background: #e0e0e0; border-radius: 15px; height: 30px; margin: 1rem 0;">
                <div class="confidence-bar" style="width: {pred['confidence']*100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top-5 chart
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top 5 Predictions")
                fig = create_confidence_chart(result["top5"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Details")
                st.metric("Inference Time", f"{inference_time*1000:.0f} ms")
                st.metric("Domain", pred["domain"].replace("_", " ").title())
                
                st.divider()
                st.caption(f"File: {uploaded_file.name}")
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


# ─── Mode 2: Record Live ───
elif mode == "Record Live":
    st.header("🎤 Record Live Audio")
    
    duration = st.slider(
        "Recording duration (seconds)",
        min_value=2,
        max_value=15,
        value=5,
        help="How long to record from your microphone"
    )
    
    st.info("🎤 Make sure your microphone is connected and allowed in browser permissions.")
    
    # Use streamlit-audiorecorder or similar component
    # For simplicity, we use a button that triggers recording
    if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
        
        # Check if sounddevice is available
        try:
            import sounddevice as sd
            
            # Countdown
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for i in range(3, 0, -1):
                status_text.markdown(f"### Starting in {i}...")
                time.sleep(0.5)
            
            # Record
            status_text.markdown("### 🔴 Recording... Make a sound!")
            
            audio = sd.rec(
                int(duration * 16000),
                samplerate=16000,
                channels=1,
                dtype="float32",
            )
            
            for i in range(duration):
                time.sleep(1)
                progress_bar.progress((i + 1) / duration)
            
            sd.wait()
            progress_bar.progress(100)
            status_text.markdown("### ✅ Recording complete!")
            
            # Playback
            st.audio(audio, sample_rate=16000)
            
            # Save to temp
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, 16000)
                temp_path = f.name
            
            # Classify
            with st.spinner("Classifying..."):
                result = st.session_state.classifier.classify(
                    temp_path,
                    domain=domain if domain != "auto" else "animal",
                )
            
            # Add to history
            st.session_state.history.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "filename": f"recording_{len(st.session_state.history)+1}.wav",
                "result": result,
            })
            
            # Display results
            pred = result["prediction"]
            emoji = get_domain_emoji(pred["domain"])
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>{emoji} {pred['class_name'].replace('_', ' ').title()}</h2>
                <h1 style="font-size: 3rem;">{pred['confidence']:.1%}</h1>
                <p style="opacity: 0.8;">confidence</p>
                <span class="domain-badge domain-{pred['domain']}">{pred['domain'].replace('_', ' ')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Top predictions
            fig = create_confidence_chart(result["top5"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
        except ImportError:
            st.error("sounddevice not installed. Run: pip install sounddevice")
            st.info("Alternatively, use the 'Upload File' mode to classify pre-recorded audio.")


# ─── Mode 3: Batch Process ───
elif mode == "Batch Process":
    st.header("📦 Batch Process Files")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        help="Upload multiple files for batch classification"
    )
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} files selected")
        
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name} ({i+1}/{len(uploaded_files)})")
                
                # Save to temp
                with tempfile.NamedTemporaryFile(suffix=Path(file.name).suffix, delete=False) as f:
                    f.write(file.read())
                    temp_path = f.name
                
                # Classify
                result = st.session_state.classifier.classify(
                    temp_path,
                    domain=domain if domain != "auto" else "animal",
                )
                result["filename"] = file.name
                results.append(result)
                
                # Add to history
                st.session_state.history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "filename": file.name,
                    "result": result,
                })
                
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            st.session_state.batch_results = results
        
        # Show batch results
        if st.session_state.batch_results:
            results = st.session_state.batch_results
            
            # Summary
            st.subheader("📊 Batch Summary")
            
            domains = [r["prediction"]["domain"] for r in results]
            domain_counts = {d: domains.count(d) for d in set(domains)}
            
            col1, col2 = st.columns(2)
            with col1:
                fig = create_domain_pie(domain_counts)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_conf = np.mean([r["prediction"]["confidence"] for r in results])
                st.metric("Files Processed", len(results))
                st.metric("Average Confidence", f"{avg_conf:.1%}")
                st.metric("Unique Predictions", len(set(r["prediction"]["class_name"] for r in results)))
            
            # Results table
            st.subheader("📋 Detailed Results")
            df_data = []
            for r in results:
                pred = r["prediction"]
                df_data.append({
                    "File": r.get("filename", r["filepath"].split("/")[-1]),
                    "Prediction": pred["class_name"].replace("_", " ").title(),
                    "Confidence": f"{pred['confidence']:.1%}",
                    "Domain": pred["domain"].replace("_", " ").title(),
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                "classification_results.csv",
                "text/csv",
                use_container_width=True,
            )


# ─── Mode 4: History ───
elif mode == "History":
    st.header("📜 Classification History")
    
    if not st.session_state.history:
        st.info("No classifications yet. Upload a file, record audio, or batch process to get started!")
    else:
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        # Show history
        for i, entry in enumerate(reversed(st.session_state.history)):
            result = entry["result"]
            pred = result["prediction"]
            
            with st.container():
                cols = st.columns([1, 3, 1, 1])
                
                with cols[0]:
                    emoji = get_domain_emoji(pred["domain"])
                    st.markdown(f"### {emoji}")
                
                with cols[1]:
                    st.markdown(f"**{pred['class_name'].replace('_', ' ').title()}**")
                    st.caption(f"{entry['filename']} · {entry['timestamp']}")
                
                with cols[2]:
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
                
                with cols[3]:
                    st.markdown(f'<span class="domain-badge domain-{pred["domain"]}">{pred["domain"].replace("_", " ")}</span>', unsafe_allow_html=True)
                
                # Expand for details
                with st.expander("Show details"):
                    fig = create_confidence_chart(result["top5"])
                    st.plotly_chart(fig, use_container_width=True)
            
            if i < len(st.session_state.history) - 1:
                st.divider()


# ─── Footer ───
st.divider()
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.8rem;'>"
    "Cross-Species Vocalization Classifier · Fine-tuned Whisper · PyTorch · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)