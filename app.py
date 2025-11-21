# ---------------------- MUST BE FIRST STREAMLIT COMMAND ----------------------
import streamlit as st

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

# ---------------------- IMPORTS ----------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ---------------------- LOAD PRETRAINED MODEL ----------------------
@st.cache_resource
def load_model():
    model_name = "bhadresh-savani/bert-base-go-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------- PREDICTION FUNCTION ----------------------
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    predicted_id = np.argmax(scores)
    emotion = model.config.id2label[predicted_id]
    return emotion

# ---------------------- DASHBOARD UI ----------------------
st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>🧠 Emotion Detection App</h1>
    <p style='text-align: center; font-size: 18px;'>Type a sentence below and click "Analyze Emotion"</p>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter your sentence here:")

emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}
color_dict = {
    "joy": "#00FF7F",
    "sadness": "#1E90FF",
    "anger": "#FF4500",
    "fear": "#8A2BE2",
    "surprise": "#FFD700",
    "disgust": "#8B0000",
    "neutral": "#708090"
}

# ---------------------- BUTTON & PREDICTION ----------------------
if st.button("Analyze Emotion"):
    if user_input.strip() != "":
        prediction = predict_emotion(user_input)

        st.markdown(f"""
            <h2 style='text-align: center; color: {color_dict.get(prediction, "#000")}'>
                {emoji_dict.get(prediction, "🤔")} {prediction.upper()}
            </h2>
        """, unsafe_allow_html=True)

        # Save history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append((user_input, prediction))
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# ---------------------- SHOW HISTORY ----------------------
if "history" in st.session_state and st.session_state["history"]:
    st.markdown("---")
    st.markdown("### 🔹 Prediction History")
    for i, (text, emotion) in enumerate(reversed(st.session_state["history"]), 1):
        st.markdown(f"**{i}. Sentence:** {text}  →  **Emotion:** {emotion}")
