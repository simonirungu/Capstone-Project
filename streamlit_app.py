
import streamlit as st
import joblib
import numpy as np
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import io
import requests
import json
import os
import re
import time

def set_calm_background():
    """Set calm and serene background for the app"""
    css = '<style>.stApp { background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 50%, #E6F3FF 100%); background-attachment: fixed; } .main .block-container { background-color: rgba(255, 255, 255, 0.92); border-radius: 20px; padding: 2.5rem; margin-top: 1rem; border: 2px solid #2E8B57; box-shadow: 0 8px 25px rgba(0,0,0,0.08); backdrop-filter: blur(10px); } h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; font-weight: 800 !important; } p, div, span, label { color: #2d2d2d !important; font-weight: 600 !important; } .stTextInput input, .stTextArea textarea { color: #1a1a1a !important; font-weight: 600 !important; background-color: rgba(255, 255, 255, 0.95) !important; border: 2px solid #2E8B57 !important; border-radius: 10px !important; } .stSuccess, .stInfo, .stWarning, .stError { font-weight: 700 !important; color: #1a1a1a !important; } .css-1d391kg { background-color: rgba(255, 255, 255, 0.98) !important; } @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } } @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } } @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }</style>'
    st.markdown(css, unsafe_allow_html=True)

# Set calm background
set_calm_background()

# ===== ENHANCED HEADER WITH ANIMATION =====
header_css = '<style>.afyamind-header { text-align: center; padding: 2rem; background: linear-gradient(135deg, #2E8B57, #32CD32, #228B22); border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.15); border: 3px solid #006400; animation: fadeInUp 1s ease-out; position: relative; overflow: hidden; } .afyamind-header::before { content: ""; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent); transform: rotate(45deg); animation: shimmer 3s infinite; } @keyframes shimmer { 0% { transform: translateX(-100%) rotate(45deg); } 100% { transform: translateX(100%) rotate(45deg); } } .afyamind-title { color: #FFFFFF !important; font-size: 4em !important; font-weight: 900 !important; margin-bottom: 0.3rem !important; text-shadow: 4px 4px 8px rgba(0,0,0,0.3); letter-spacing: 2px; position: relative; z-index: 2; } .afyamind-subtitle { color: #F0FFF0 !important; font-size: 1.6em !important; font-weight: 700 !important; margin-top: 0 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); position: relative; z-index: 2; } .stButton>button { background: linear-gradient(45deg, #2E8B57, #32CD32) !important; color: white !important; font-weight: 700 !important; border: none !important; border-radius: 12px !important; padding: 12px 24px !important; transition: all 0.3s ease !important; animation: pulse 2s infinite; box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3) !important; } .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4) !important; animation: none !important; } .stChatMessage { animation: slideIn 0.5s ease-out !important; } .stSpinner > div { border-color: #2E8B57 transparent transparent transparent !important; }</style>'

st.markdown(header_css, unsafe_allow_html=True)
st.markdown('<div class="afyamind-header"><h1 class="afyamind-title">🌿 AfyaMind</h1><p class="afyamind-subtitle">Kenyan Emotions Classifier and Mental Health Assistant</p></div>', unsafe_allow_html=True)

# Define the custom class first
class EmotionClassifierSKLearn:
    def __init__(self, model_path=None, max_len=128, threshold=0.5):
        self.model_path = model_path
        self.max_len = max_len
        self.threshold = threshold
        self.classes_ = np.array(['anger_mapped', 'neutral_mapped', 'joy_mapped', 'surprise_mapped', 
                                 'sadness_mapped', 'disgust_mapped', 'fear_mapped'])

    def fit(self, X, y=None):
        return self

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, texts):
        if not hasattr(self, 'model'):
            self._load_model()

        if isinstance(texts, str): texts = [texts]
        all_probs = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer.encode_plus(text, truncation=True, max_length=self.max_len, 
                                                   padding='max_length', return_token_type_ids=True)
                ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
                mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
                token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(self.device)

                outputs = self.model(ids, mask, token_type_ids)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
                all_probs.append(probs)

        return np.array(all_probs)

    def predict_emotions(self, texts):
        preds = self.predict(texts)
        return [[self.classes_[i] for i, p in enumerate(pred) if p == 1] for pred in preds]

    def _load_model(self):
        class BERTClass(torch.nn.Module):
            def __init__(self):
                super(BERTClass, self).__init__()
                self.roberta = AutoModel.from_pretrained('roberta-base')
                self.fc = torch.nn.Linear(768, 7)
            def forward(self, ids, mask, token_type_ids):
                _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
                return self.fc(features)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BERTClass()

        # Load model from Hugging Face Hub
        try:
            # Add loading animation
            with st.spinner("🔄 Downloading AI model from Hugging Face... This may take a moment."):
                model_path = hf_hub_download(
                    repo_id="simonirungu/AfyaMind",
                    filename="bert_model_weights.pth",
                    cache_dir="./model_cache"
                )
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            st.success("✅ Model loaded successfully from Hugging Face!")
        except Exception as e:
            st.error(f"❌ Error loading model from Hugging Face: {e}")
            # Fallback: try local file
            try:
                if os.path.exists('bert_model_weights.pth'):
                    self.model.load_state_dict(torch.load('bert_model_weights.pth', map_location='cpu'))
                    st.info("ℹ️ Model loaded from local file")
                else:
                    st.error("❌ Model file not found locally either")
                    return
            except Exception as e2:
                st.error(f"❌ Error loading local model: {e2}")
                return

        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Load emotion model with caching
@st.cache_resource
def load_emotion_model():
    return EmotionClassifierSKLearn()

# Initialize model
model = load_emotion_model()

# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def enhance_emotion_detection(text, base_emotions):
    """Enhance emotion detection with Kenyan context"""
    text_lower = text.lower()

    # Kenyan context indicators
    sadness_indicators = ['nairobi traffic', 'matatu', 'high cost of living', 'hustle', 'siasa']
    anger_indicators = ['corruption', 'inflation', 'nairobi traffic jam', 'county government']
    fear_indicators = ['insecurity', 'nairobi cbd', 'flooding', 'economic uncertainty']
    joy_indicators = ['safaricom', 'm-pesa', 'nyama choma', 'beautiful kenya', 'magical kenya']

    # Check for Kenyan context
    if any(indicator in text_lower for indicator in sadness_indicators):
        if 'sadness_mapped' not in base_emotions:
            base_emotions.append('sadness_mapped')
    if any(indicator in text_lower for indicator in anger_indicators):
        if 'anger_mapped' not in base_emotions:
            base_emotions.append('anger_mapped')
    if any(indicator in text_lower for indicator in fear_indicators):
        if 'fear_mapped' not in base_emotions:
            base_emotions.append('fear_mapped')
    if any(indicator in text_lower for indicator in joy_indicators):
        if 'joy_mapped' not in base_emotions:
            base_emotions.append('joy_mapped')

    return base_emotions

def get_deepseek_response(user_message, detected_emotions):
    """Get mental health response from DeepSeek API"""

    api_key = st.session_state.get('sk-59764f2289ea4ba99dbf5abcf7129114', '')
    if not api_key:
        return "Please configure your API key in the sidebar first."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"""
    You are AfyaMind, a compassionate mental health assistant for Kenyans. 

    User's message: "{user_message}"
    Detected emotions: {", ".join(detected_emotions)}

    Provide a warm, personalized mental health response that:
    1. Genuinely addresses their specific situation and emotions
    2. Uses Cognitive Behavioral Therapy (CBT) principles
    3. Provides practical, actionable advice
    4. Shows empathy and understanding
    5. Includes Kenyan cultural context where relevant
    6. Mentions Kenyan mental health resources

    Make the response feel personal and directly relevant to their situation.
    Keep it conversational and supportive.
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are AfyaMind, a Kenyan mental health assistant. You provide personalized, evidence-based mental health support using CBT principles. You are warm, empathetic, and practical. You adapt your responses to each user's specific situation and emotions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 1000
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API Error {response.status_code}. Please check your API key."
    except Exception as e:
        return f"Connection error: {str(e)}"

def get_fallback_response(user_message, detected_emotions):
    """Fallback responses when API is not available"""
    primary_emotion = detected_emotions[0].lower() if detected_emotions else 'neutral'

    responses = {
        'sadness': "I hear that you're feeling down, and I want you to know your feelings are completely valid. When we experience sadness, it often signals that something important needs attention. You might try starting with small activities - even a short walk or reaching out to a friend can help shift your mood. Remember, emotions come and go like waves.",
        'anger': "I can sense the frustration in your words. Anger often arises when we feel our boundaries have been crossed. Try taking a moment to breathe deeply before responding. This creates space between the trigger and your reaction. Your anger might be pointing toward something that needs to change.",
        'fear': "I hear the worry in what you're sharing. These feelings are your body's way of trying to protect you. You might try grounding techniques - name five things you can see, four you can touch, three you can hear. This can help bring you back to the present moment.",
        'joy': "It's wonderful to hear about these positive experiences! Savoring joyful moments actually builds psychological resources. Consider sharing your happiness with others or writing down what made this moment special.",
        'neutral': "Thank you for sharing what's on your mind. Sometimes neutral emotional spaces are valuable for reflection and recharging. This could be a good time for mindfulness practice or considering what routines support your wellbeing."
    }

    base_response = responses.get(primary_emotion, "Thank you for sharing your experience. Your feelings are valid and worthy of attention. Consider what small step might support your wellbeing today.")

    return f"""{base_response}

**Kenyan Mental Health Resources:**
• Kenya Red Cross Psychological Support: 1199
• Nairobi Women's Hospital Gender Violence Recovery Centre: 0800 720 544
• Befrienders Kenya: +254 722 178 177
• Emergency Services: 112/999

**Disclaimer:** I am a mental health assistant and not a licensed therapist. My suggestions are based on psychological research and should not replace professional medical advice.
"""

def get_enhanced_emotions(text):
    """Get emotions with Kenyan context detection"""
    if model is None:
        return []

    try:
        base_emotions = model.predict_emotions([text])[0]
        enhanced_emotions = enhance_emotion_detection(text, base_emotions.copy())
        return enhanced_emotions
    except Exception as e:
        st.error(f"Error predicting emotions: {e}")
        return []

def get_mental_health_response(user_message, detected_emotions, use_api=True):
    """Get mental health response"""
    api_key = st.session_state.get('deepseek_api_key', '')

    # Check if API should be used and key is valid
    should_use_api = use_api and api_key and api_key.strip() and api_key != "YOUR_DEEPSEEK_API_KEY_HERE"

    if should_use_api:
        api_response = get_deepseek_response(user_message, detected_emotions)
        # Only use API response if it doesn't contain error messages
        if not any(error in api_response.lower() for error in ['api error', 'connection error', 'please check']):
            return api_response

    # Fallback to rule-based responses
    return get_fallback_response(user_message, detected_emotions)

# Check if model is loaded
if not hasattr(model, 'model') or model.model is None:
    st.error("❌ Emotion classifier model could not be loaded. Some features may not work properly.")
else:
    st.success("✅ Emotion model loaded successfully!")

# Sidebar configuration with enhanced styling
with st.sidebar:
    sidebar_css = '<style>.sidebar-header { text-align: center; background: linear-gradient(135deg, #2E8B57, #32CD32, #228B22); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1.5rem; box-shadow: 0 6px 20px rgba(0,0,0,0.15); border: 2px solid #006400; animation: fadeInUp 1s ease-out; } .stSidebar h3 { color: #1a1a1a !important; font-weight: 800 !important; } .stSidebar label { color: #2d2d2d !important; font-weight: 700 !important; } .stTextInput input, .stTextArea textarea { color: #1a1a1a !important; font-weight: 600 !important; background-color: rgba(255, 255, 255, 0.95) !important; border: 2px solid #2E8B57 !important; border-radius: 8px !important; }</style>'

    st.markdown(sidebar_css, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header"><h2 style="margin: 0; color: white; font-weight: 800;">🌿 AfyaMind</h2><p style="margin: 0; font-size: 0.9em; opacity: 0.95; font-weight: 600;">Mental Wellness</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("API Configuration")
    use_api = st.sidebar.checkbox("Use DeepSeek API for Enhanced Responses", value=True)

    if 'deepseek_api_key' not in st.session_state:
        st.session_state.deepseek_api_key = ""

    api_key_input = st.sidebar.text_input("DeepSeek API Key:", 
                                         value="",
                                         type="password",
                                         placeholder="Enter your API key here")

    if api_key_input:
        st.session_state.deepseek_api_key = api_key_input
        if api_key_input.strip() and api_key_input != "YOUR_DEEPSEEK_API_KEY_HERE":
            st.sidebar.success("✅ API Key configured!")
        else:
            st.sidebar.warning("⚠️ Please enter a valid API key")

    if use_api and (not st.session_state.deepseek_api_key or st.session_state.deepseek_api_key == "YOUR_DEEPSEEK_API_KEY_HERE"):
        st.sidebar.warning("🔑 Enter DeepSeek API key for enhanced responses")
        st.sidebar.info("🌐 Get free API key: https://platform.deepseek.com/")

    # Navigation
    page = st.sidebar.selectbox("Choose a tool:", 
                               ["Single Text Analysis", "Batch Analysis", "Mental Health Chatbot", "CSV Intervention Generator"])

if page == "Single Text Analysis":
    st.header("📊 Single Text Emotion Analysis")
    text = st.text_input("Enter text to analyze:", placeholder="Type your message here...")

    if st.button("Analyze Emotion 🔍"):
        if text.strip():
            with st.spinner("🔍 Analyzing emotions..."):
                time.sleep(1)  # Add slight delay for animation
                emotions = get_enhanced_emotions(text)
            if emotions:
                emotion_names = [e.replace('_mapped', '').title() for e in emotions]
                st.success(f"🎯 Detected Emotions: {', '.join(emotion_names)}")
            else:
                st.info("🤔 No strong emotions detected in the text.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

elif page == "Batch Analysis":
    st.header("📈 Batch Text Analysis")

    st.subheader("Option 1: Enter multiple texts")
    batch_texts = st.text_area("Enter texts (one per line):", height=150)

    st.subheader("Option 2: Upload CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file with a 'text' column", type="csv")

    if st.button("Analyze Batch 📊"):
        results = []

        # Process text area input
        if batch_texts.strip():
            texts = [t.strip() for t in batch_texts.split('\n') if t.strip()]
            for i, text in enumerate(texts):
                emotions = get_enhanced_emotions(text)
                if emotions:
                    emotion_names = [e.replace('_mapped', '').title() for e in emotions]
                    results.append(f"Text {i+1}: {', '.join(emotion_names)}")
                else:
                    results.append(f"Text {i+1}: No emotions detected")

        # Process CSV file input - FIXED
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].astype(str).dropna().tolist()
                    enhanced_emotions = []

                    for text in texts:
                        emotions = get_enhanced_emotions(text)
                        enhanced_emotions.append(emotions)

                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Text': texts,
                        'Detected_Emotions': [", ".join([e.replace('_mapped', '').title() for e in emotion_list]) if emotion_list else "None" 
                                             for emotion_list in enhanced_emotions]
                    })

                    st.subheader("Analysis Results")
                    st.dataframe(results_df)

                    # Download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"❌ Error processing CSV file: {e}")

        # Display text area results
        if results:
            st.subheader("Analysis Results")
            for result in results:
                st.write(result)

elif page == "Mental Health Chatbot":
    st.header("💬 Mental Health Chatbot")

    # Show API status
    api_key = st.session_state.get('deepseek_api_key', '')
    api_enabled = use_api and api_key and api_key.strip() and api_key != "YOUR_DEEPSEEK_API_KEY_HERE"

    if api_enabled:
        st.success("✅ Using DeepSeek API for personalized responses")
    else:
        st.info("ℹ️ Using built-in responses (configure API for personalized responses)")

    st.info("ℹ️ About AfyaMind: I'm here to provide supportive mental health conversations using evidence-based strategies.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("💭 Understanding your emotions and preparing response..."):
                time.sleep(1)  # Add slight delay for animation
                emotions = get_enhanced_emotions(prompt)
                emotion_names = [e.replace('_mapped', '').title() for e in emotions] if emotions else ["Neutral"]

                response = get_mental_health_response(prompt, emotion_names, use_api)

                st.markdown(f"**🎭 Detected emotions:** {', '.join(emotion_names)}")
                st.markdown("---")
                st.markdown(response)

                full_response = f"**🎭 Detected emotions:** {', '.join(emotion_names)}\n\n{response}"
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

elif page == "CSV Intervention Generator":
    st.header("📋 CSV Intervention Generator")

    st.write("Upload a CSV file with emotional texts to generate personalized mental health interventions.")

    uploaded_file = st.file_uploader("Upload CSV with 'text' column", type="csv", key="intervention_csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                sample_df = df.head(5)  # Increased limit for better testing

                if st.button("Generate Interventions 🚀"):
                    interventions = []

                    for idx, row in sample_df.iterrows():
                        text = str(row['text'])
                        emotions = get_enhanced_emotions(text)
                        emotion_names = [e.replace('_mapped', '').title() for e in emotions] if emotions else ["Neutral"]

                        intervention = get_mental_health_response(text, emotion_names, use_api)
                        interventions.append({
                            'Original_Text': text,
                            'Detected_Emotions': ", ".join(emotion_names),
                            'AI_Intervention': intervention
                        })

                    # Create results dataframe
                    results_df = pd.DataFrame(interventions)
                    st.subheader("Generated Interventions")
                    st.dataframe(results_df)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Interventions as CSV",
                        data=csv,
                        file_name="mental_health_interventions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ CSV file must contain a 'text' column")
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {e}")

# Disclaimer
st.markdown("---")
st.markdown("**⚠️ Important Disclaimer:** AfyaMind is an AI assistant designed to provide general mental health support and information. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified mental health providers.")
