
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load the sentiment model
with open('sentiment_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("Sentiment Analysis with Loaded Model")

# Input text area for user input
text_input = st.text_area("Enter text for sentiment analysis", "please insert text")

# Button to trigger prediction
if st.button("Analyze Sentiment"):
    # Get predictions from the loaded model
    predictions = loaded_model.predict([text_input])
    
    # Display the result
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Prediction: {predictions[0]}")
