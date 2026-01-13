import streamlit as st
import numpy as np
import pickle
import speech_recognition as sr
import pyttsx3
import requests
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model and tokenizer
model = load_model('next_word_lstm.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize translator and text-to-speech engine
translator = Translator()
tts_engine = pyttsx3.init()

# Function to predict the next words
def predict_next_words(model, tokenizer, text, max_sequence_len, num_words=3):
    predicted_words = []
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                text += " " + word  # Append predicted word to the text
                predicted_words.append(word)
                break
    return predicted_words

# Function to fetch word definition from an API
def get_word_definition(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        definition = data[0]["meanings"][0]["definitions"][0]["definition"]
        return definition
    return "Definition not found."

# Function to capture voice input
def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening... Speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the speech."
    except sr.RequestError:
        return "Speech service unavailable."

# Function to speak out text
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to translate text
def translate_text(text, target_language):
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Streamlit UI
st.set_page_config(page_title="Next Word Prediction", layout="wide", initial_sidebar_state="expanded")

# Theme Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: white; }
        .stTextInput, .stButton, .stSelectbox { background-color: #333; color: white; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🔮 Next Word Prediction with LSTM")

# Speech-to-Text Button
if st.button("🎙️ Use Voice Input"):
    spoken_text = get_speech_input()
    if spoken_text:
        st.session_state["text_input"] = spoken_text
        st.rerun()

# Text Input Field
input_text = st.text_input("✍️ Enter a sequence of words:", key="text_input")

# Number of Words to Predict
num_words = st.slider("🔢 Number of words to predict:", min_value=1, max_value=5, value=3)

# Language Selection for Translation
languages = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi"}
selected_language = st.selectbox("🌍 Translate Prediction To:", list(languages.keys()))

# Predict Button
if st.button("🔮 Predict Next Words"):
    if not input_text:
        st.warning("⚠️ Please enter a sequence of words.")
    else:
        max_sequence_len = model.input_shape[1]
        predicted_words = predict_next_words(model, tokenizer, input_text, max_sequence_len, num_words)

        if predicted_words:
            predicted_sentence = " ".join(predicted_words)
            st.success(f"✨ Predicted words: **{predicted_sentence}**")

            # Speak Prediction
            if st.button("🔊 Speak Prediction"):
                speak_text(predicted_sentence)

            # Get Definitions
            with st.expander("📖 Word Definitions"):
                for word in predicted_words:
                    definition = get_word_definition(word)
                    st.write(f"**{word.capitalize()}**: {definition}")

            # Translate Prediction
            translated_text = translate_text(predicted_sentence, languages[selected_language])
            st.write(f"🌍 Translated ({selected_language}): **{translated_text}**")

        else:
            st.error("❌ No prediction could be made.")
