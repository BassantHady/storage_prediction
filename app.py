"""
app.py
======
Simple Streamlit web application for storage type prediction.
Provides an interactive UI to test the models.

Author: NLP Engineering Team

Usage:
    streamlit run app.py
"""

import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import streamlit as st
import requests
import json
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Storage Type Predictor",
    page_icon="🧊",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"
API_ENABLED = False  # Set to True when API is running


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def predict_via_api(text: str, model: str) -> Dict:
    """Send prediction request to API."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text, "model": model}
    )
    return response.json()


def predict_local(text: str, model_name: str) -> Dict:
    """Make prediction using local model (without API)."""
    from predict import StoragePredictor
    
    if "predictor" not in st.session_state:
        st.session_state.predictor = {}
    
    if model_name not in st.session_state.predictor:
        with st.spinner(f"Loading {model_name} model..."):
            st.session_state.predictor[model_name] = StoragePredictor(model_name=model_name)
    
    predictor = st.session_state.predictor[model_name]
    return predictor.predict(text)


def main():
    st.title("🧊 Storage Type Predictor")
    st.markdown("""
    This app predicts whether an item should be stored in the **freezer**, **fridge**, 
    or at **room temperature** based on a natural language question.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection (removed lstm)
        model_options = ["distilbert", "logistic", "svm", "rf"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            help="DistilBERT is the most accurate but slower. Logistic Regression is fastest."
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        - **Freezer**: Items that need freezing (-18°C)
        - **Fridge**: Items that need refrigeration (2-4°C)
        - **Normal**: Items safe at room temperature
        """)
        
        st.markdown("---")
        st.markdown("### Example Questions")
        examples = [
            "Should I store milk in the fridge?",
            "Where should I keep frozen chicken nuggets?",
            "Can I leave my laptop at room temperature?",
            "Does yogurt need to be refrigerated?",
            "Is it safe to freeze fresh salmon?",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.example_text = ex
    
    # Main input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if "example_text" in st.session_state:
            default_text = st.session_state.example_text
            del st.session_state.example_text
        else:
            default_text = "Should I store milk in the fridge?"
        
        user_input = st.text_area(
            "Enter your question:",
            value=default_text,
            height=100,
            placeholder="e.g., 'Where should I keep frozen pizza?'"
        )
    
    with col2:
        st.markdown("### ")
        st.markdown("### ")
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    # Prediction
    if predict_button and user_input:
        with st.spinner("Predicting..."):
            try:
                result = predict_local(user_input, selected_model)
                
                # Display result
                st.markdown("---")
                st.header("Prediction Result")
                
                # Color-coded result
                storage = result["predicted_storage"]
                if storage == "freezer":
                    color = "#3498db"
                    emoji = "❄️"
                elif storage == "fridge":
                    color = "#2ecc71"
                    emoji = "🧊"
                else:
                    color = "#f39c12"
                    emoji = "🏠"
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="margin: 0;">{emoji} {storage.upper()} {emoji}</h1>
                    <p style="font-size: 18px; margin-top: 10px;">Confidence: {result['confidence']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities
                if "all_probs" in result and result["all_probs"]:
                    st.subheader("Confidence Scores")
                    col_a, col_b, col_c = st.columns(3)
                    
                    probs = result["all_probs"]
                    with col_a:
                        st.metric("❄️ Freezer", f"{probs.get('freezer', 0):.2%}")
                    with col_b:
                        st.metric("🧊 Fridge", f"{probs.get('fridge', 0):.2%}")
                    with col_c:
                        st.metric("🏠 Normal", f"{probs.get('normal', 0):.2%}")
                
                # Show input text
                with st.expander("View Input Text"):
                    st.write(result["text"])
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure models are trained first. Run `python src/train.py` to train models.")
    
    elif predict_button and not user_input:
        st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit, FastAPI, and Transformers | "
        "Predicts freezer/fridge/normal storage types</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()