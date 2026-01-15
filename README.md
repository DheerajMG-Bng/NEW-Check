LSTM-Based Next Word Prediction

An end-to-end NLP deep learning project that predicts the next most probable word(s) given an input text sequence using an LSTM neural network.
The model is trained on literary text and deployed as a live web application.
----------------------------------------------------------------------------------------------------------------------------------
Live Demo

🔗 Deployed Application:
--> https://lstm-next-word-prediction-rnn-dheeraj-m-g.streamlit.app/


<img width="1919" height="1079" alt="Screenshot 2026-01-15 122910" src="https://github.com/user-attachments/assets/ed4122cc-a672-4990-880e-be3ae5cfbc28" />


 Project Objective

Dataset

Source: NLTK Gutenberg Corpus

Text Used: Shakespeare’s Hamlet

Task: Next-word language modeling

Workflow

Text preprocessing (tokenization, sequencing, padding)

N-gram sequence generation

LSTM-based language model training

Model serialization using TensorFlow SavedModel

Cloud deployment using Streamlit

Model Architecture

Embedding Layer

LSTM Layer

Dense + Softmax Output

Web Application

User text input

Predicts next 1–5 words

Real-time inference

Optimized model loading with caching

Deployment

Platform: Streamlit Community Cloud

Model Format: TensorFlow SavedModel

Key Learning: SavedModel is more reliable than .h5 for cloud deployment

📚Concepts Learned

NLP preprocessing & tokenization

Sequence modeling with LSTM

Language modeling

Model serialization strategies

Streamlit app development

Real-world ML deployment & debugging

Author

Dheeraj M G
Computer Science Undergraduate
