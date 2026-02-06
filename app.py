import streamlit as st
import sys
import os
import random

# Add src to path
sys.path.append(os.path.abspath("src"))
from predict import SentimentPredictor

# Page Config
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Aesthetics
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #2a2a40;
        color: white;
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea {
        background-color: #2a2a40;
        color: white;
        border-radius: 10px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(229, 46, 113, 0.4);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Header
st.title("🛍️ Flipkart Product Review Sentiment Analysis")
st.markdown("### 🤖 AI-Powered Sentiment Classifier")
st.markdown("Enter a product review below to detect whether it's **Positive** or **Negative**.")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.info("Model loaded from **MLflow Registry** (Production Stage)")

# Load Model (Cached)
@st.cache_resource
def get_predictor():
    return SentimentPredictor(use_local_artifacts=True)

try:
    predictor = get_predictor()
    st.sidebar.success("Model Loaded Successfully! ✅")
except Exception as e:
    st.sidebar.error("Error loading model. Make sure to run training first.")
    st.error(f"Details: {e}")
    st.stop()

# Sample Reviews
st.sidebar.subheader("Test with Samples")
samples = [
    "Amazing product! The quality is top notch and delivery was super fast.",
    "Worst experience ever. The product broke after one day.",
    "Decent for the price, but could be better.",
    "Absolutely love it! Highly recommended.",
    "Don't buy this trash. Waste of money."
]

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

def set_sample(text):
    st.session_state.review_input = text

for sample in samples:
    st.sidebar.button(sample[:30] + "...", on_click=set_sample, args=(sample,))

# Input Section
user_input = st.text_area("Review Text", value=st.session_state.review_input, height=150, placeholder="Type your review here...")

if st.button("Analyze Sentiment 🚀"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = predictor.predict(user_input)
            
            sentiment = result['prediction']
            prob = result['probability']
            
            # Display Result
            if sentiment == "Positive":
                color = "#4CAF50" # Green
                emoji = "😊"
                msg = "Positive Feedback"
            else:
                color = "#FF5252" # Red
                emoji = "😠"
                msg = "Negative Feedback"
                
            st.markdown(f"""
                <div class="result-card" style="border-left: 5px solid {color};">
                    <h1 style="color: {color}; margin: 0;">{emoji} {sentiment}</h1>
                    <p style="font-size: 1.2rem; opacity: 0.8;">{msg}</p>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 10px; margin-top: 15px;">
                        <strong>Confidence Score:</strong>
                        <div style="width: 100%; background: #333; border-radius: 5px; height: 10px; margin-top: 5px;">
                            <div style="width: {prob*100}%; background: {color}; height: 100%; border-radius: 5px;"></div>
                        </div>
                        <p style="margin-top: 5px;">{prob*100:.2f}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("Please enter some text first.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, MLflow, and Scikit-Learn")
