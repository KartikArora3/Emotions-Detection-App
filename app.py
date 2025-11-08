import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ----------------------
# Step 1: Dataset
# ----------------------
data = {
    "text": [
        "I am very happy today",
        "This is so sad",
        "I am angry at you",
        "I feel great and joyful",
        "I am scared of the dark",
        "That movie was terrifying",
        "What a surprise party!",
        "I am feeling wonderful",
        "This makes me cry",
        "I am so mad right now",
        "I am thrilled to see you",
        "I feel depressed and alone",
        "He yelled at me angrily",
        "That was a shocking event",
        "I love this so much",
        "I hate everything today",
        "I am nervous about tomorrow",
        "This is the best day ever",
        "I’m disappointed in myself",
        "She was frightened by the noise"
    ],
    "emotion": [
        "happy", "sad", "angry", "happy", "fear", "fear", "surprise", "happy", 
        "sad", "angry", "happy", "sad", "angry", "surprise", "happy", 
        "angry", "fear", "happy", "sad", "fear"
    ]
}

df = pd.DataFrame(data)

# ----------------------
# Step 2: Preprocess
# ----------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'#\w+','', text)
    text = re.sub(r'[^a-z\s]','',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(preprocess_text)

# ----------------------
# Step 3: Tokenizer & Labels
# ----------------------
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['emotion'])

# ----------------------
# Step 4: LSTM Model
# ----------------------
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=50, verbose=0)

# ----------------------
# Step 5: Predictor
# ----------------------
def predictor(text):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    pred = model.predict(padded, verbose=0)
    return label_encoder.inverse_transform([np.argmax(pred)])[0]

# ----------------------
# Step 6: Streamlit UI
# ----------------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>🧠 Emotion Detection App</h1>
    <p style='text-align: center; font-size: 18px;'>Type a sentence below and click "Analyze Emotion"</p>
""", unsafe_allow_html=True)

# Input box
user_input = st.text_area("Enter your sentence here:")

# Button & Prediction
if st.button("Analyze Emotion"):
    if user_input.strip() != "":
        prediction = predictor(user_input)

        # Display with color & emoji
        emoji_dict = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😡",
            "fear": "😨",
            "surprise": "😲"
        }
        color_dict = {
            "happy": "#00FF7F",
            "sad": "#1E90FF",
            "angry": "#FF4500",
            "fear": "#8A2BE2",
            "surprise": "#FFD700"
        }

        st.markdown(f"""
            <h2 style='text-align: center; color: {color_dict.get(prediction, "#000")};'>
            {emoji_dict.get(prediction, "🤔")} {prediction.upper()}
            </h2>
        """, unsafe_allow_html=True)

        # Save history in session
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append((user_input, prediction))

    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Show history
if 'history' in st.session_state and st.session_state['history']:
    st.markdown("---")
    st.markdown("### 🔹 Prediction History")
    for i, (text, pred) in enumerate(reversed(st.session_state['history']), 1):
        st.markdown(f"**{i}. Sentence:** {text}  →  **Emotion:** {pred}")
