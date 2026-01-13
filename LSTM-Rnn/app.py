import streamlit as st
import numpy as np
import pickle
import requests
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model & tokenizer (cached)
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("next_word_lstm.h5")
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

translator = Translator()

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_words(model, tokenizer, text, max_sequence_len, num_words=3):
    predicted_words = []
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                text += " " + word
                predicted_words.append(word)
                break
    return predicted_words

# -----------------------------
# Dictionary API
# -----------------------------
def get_word_definition(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data[0]["meanings"][0]["definitions"][0]["definition"]
    return "Definition not found."

# -----------------------------
# Translation
# -----------------------------
def translate_text(text, target_language):
    translated = translator.translate(text, dest=target_language)
    return translated.text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 Next Word Prediction using LSTM")

st.markdown(
    """
This application predicts the **next most probable word(s)** using a  
**pretrained LSTM language model trained on Shakespeare's Hamlet**.
"""
)

# Text input
input_text = st.text_input("✍️ Enter a sequence of words:")

# Number of words
num_words = st.slider("🔢 Number of words to predict:", 1, 5, 3)

# Translation
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
}
selected_language = st.selectbox(
    "🌍 Translate prediction to:", list(languages.keys())
)

# Predict button
if st.button("🔮 Predict Next Words"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        max_sequence_len = model.input_shape[1]
        predicted_words = predict_next_words(
            model, tokenizer, input_text, max_sequence_len, num_words
        )

        if predicted_words:
            predicted_sentence = " ".join(predicted_words)
            st.success(f"✨ Predicted words: **{predicted_sentence}**")

            # Definitions
            with st.expander("📖 Word Definitions"):
                for word in predicted_words:
                    definition = get_word_definition(word)
                    st.write(f"**{word.capitalize()}**: {definition}")

            # Translation
            translated_text = translate_text(
                predicted_sentence, languages[selected_language]
            )
            st.write(
                f"🌍 Translated ({selected_language}): **{translated_text}**"
            )
        else:
            st.error("❌ No prediction could be made.")
