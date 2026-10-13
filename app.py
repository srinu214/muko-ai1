import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import os

# Set professional medical page layout optimized for mobile responsive viewports
st.set_page_config(
    page_title="MUKO AI - Edge Triage", 
    page_icon="🩺", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Mobile-responsive viewport and touch-friendly styles
st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        h1 {
            font-size: 1.6rem !important;
        }
        .stButton>button {
            width: 100% !important;
            height: 3.2rem !important;
            font-size: 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 MUKO AI: Edge Intelligence Triage")
st.markdown('**Project Motto:** *"Giving every village the eyes to spot oral cancer before it catches them."*')
st.caption("A pure-software operational layout optimized for offline deployment on 4GB RAM devices.")

# 🧠 LOAD THE AI MODEL SAFELY
@st.cache_resource
def load_onnx_session():
    # Robust resolution across persistent root container folder maps and relative paths
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "muko_edge_processor.onnx"),
        os.path.abspath("muko_edge_processor.onnx"),
        "./muko_edge_processor.onnx",
        "muko_edge_processor.onnx"
    ]
    for candidate in search_paths:
        if os.path.exists(candidate):
            return ort.InferenceSession(candidate)
    raise FileNotFoundError(f"muko_edge_processor.onnx could not be located in: {search_paths}")

try:
    ort_session = load_onnx_session()
    st.success("✅ Core Edge Neural Engine Active")
except Exception as e:
    st.error(f"❌ Model initialization error: {e}")

# 📊 STREAM 1: ALGORITHMIC ROI GEOMETRIC SHAPE ANALYZER
def extract_roi_geometry(cv_image):
    """
    Computes mathematical shape markers using real OpenCV primitives.
    Ulcers show high solidity (smooth contours); Cancers display low solidity (jagged borders).
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 1.0, 0.0  # Default safe metrics if no distinct edge mass is found
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull) if len(hull) > 0 else area
    
    # Solidity = Contour Area / Convex Hull Area (Edge Jaggedness Biomarker)
    solidity = float(area) / hull_area if hull_area > 0 else 1.0
    
    total_pixels = cv_image.shape[0] * cv_image.shape[1]
    size_ratio = float(area) / total_pixels
    
    return solidity, size_ratio

# 📋 STREAM 2: FRONTLINE CLINICAL QUESTIONNAIRE
st.subheader("📋 Frontline Clinical Risk Taps")
with st.container(border=True):
    is_tobacco = st.checkbox("Is the patient a regular tobacco / gutkha addict?")
    is_unhealed = st.checkbox("Has the mouth sore/ulcer been unhealed for more than 2 weeks?")
    is_burning = st.checkbox("Experiencing a burning sensation when eating spicy food?")

# 📷 CAMERA / IMAGE INPUT NODE
st.subheader("📷 Capture Oral Chamber")
uploaded_file = st.file_uploader("Snap a photo using your phone or upload a snapshot", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image inside web interface frame
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Oral Chamber View", use_container_width=True)
    
    # Convert image format to match standard OpenCV processing arrays
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    if st.button("Execute Edge Diagnostics", type="primary"):
        with st.spinner("Processing structural arrays..."):
            
            # 1. Compute Algorithmic Geometry Stream
            solidity_score, size_index = extract_roi_geometry(cv_img)
            
            # 2. Compute Neural Processing Embedding Stream
            resized = cv2.resize(cv_img, (224, 224))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            float_img = resized_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (float_img - mean) / std
            
            input_tensor = np.transpose(normalized, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Run the offline forward pass model calculation
            input_name = ort_session.get_inputs()[0].name
            raw_outputs = ort_session.run(None, {input_name: input_tensor})
            logits = raw_outputs[0][0]
            
            # Binary target evaluation matrix check (Logit 1 = Cancer, Logit 0 = Normal)
            is_neural_suspicious = (logits[1] > logits[0])
            
            # 🎛️ STREAM 3: THE UNIFIED MATRIX MASTER ROUTER
            triage_status = "GREEN"
            
            # Condition Amber: Edge patterns look irregular, or symptoms imply checkup requirement
            if is_neural_suspicious or solidity_score < 0.78 or is_burning:
                triage_status = "AMBER"
                
            # Condition Red: Deep jagged edges paired with network flags OR long-term unhealed sore
            if (is_neural_suspicious and solidity_score < 0.60) or is_unhealed:
                triage_status = "RED"
                
            # Compile guidance feedback notes string
            guidance = "Routine checkup recommended. Maintain baseline oral hygiene protocols."
            if triage_status == "AMBER":
                guidance = "Non-uniform tissue variations detected. Schedule local clinic screening checkpoint in 14 days to re-test shape stability."
            elif triage_status == "RED":
                guidance = "Jagged, asymmetric tissue structures matching malignant contours verified. Immediate specialist biopsy referral required."
                
            if is_tobacco:
                guidance += " [WARNING: Patient is a regular tobacco addict. Extreme high-risk systemic environment. Immediate cessation advised.]"
                
            # 🚨 DISPLAY ULTIMATE TRIAGE STATUS BANNERS
            st.subheader("📋 System Triage Analysis Report")
            
            if triage_status == "GREEN":
                st.success(f"🟢 **GREEN - Everything is Good**\n\n**Visual Metrics:** Smoothness Index: {solidity_score:.2f} | **Guidance:** {guidance}")
            elif triage_status == "AMBER":
                st.warning(f"🟡 **AMBER - Suspicious / Retest After 2 Weeks**\n\n**Visual Metrics:** Smoothness Index: {solidity_score:.2f} | **Guidance:** {guidance}")
            elif triage_status == "RED":
                st.error(f"🔴 **RED - Urgent Danger / Cancer Prone Referral**\n\n**Visual Metrics:** Smoothness Index: {solidity_score:.2f} | **Guidance:** {guidance}")
