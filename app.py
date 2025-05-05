# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the trained model and tokenizer
model = tf.keras.models.load_model('spam_detection_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set maximum sequence length (same as during training)
max_len = 100

# Function to preprocess and predict
def predict_email(email_text):
    # Remove punctuations
    email_text = email_text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([email_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    return prediction

# Streamlit UI
st.title("📧 Email Spam Detection App")
st.write("Enter the email content below to predict if it's spam or not.")

# User input
email_text = st.text_area("✉️ Enter Email Text:", height=150)

# Prediction on button click
if st.button("🚀 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        prediction = predict_email(email_text)
        if prediction > 0.5:
            st.error(f"❗️ This email is likely **SPAM** with a probability of {prediction:.2f}")
        else:
            st.success(f"✅ This email is likely **NOT SPAM** with a probability of {1 - prediction:.2f}")

# About Section
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This Email Spam Detection App uses an LSTM model to classify emails as spam or non-spam.
""")
