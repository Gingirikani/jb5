import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

model_paths = {
    "Logistic Regression": os.path.join(base_path, "model_lr.pkl"),
    "Naive Bayes": os.path.join(base_path, "model_nb.pkl"),
    "Support Vector Machine": os.path.join(base_path, "svc_model.pkl")
}

#-----------sidebar
page = st.sidebar.selectbox('page navigator', ["predictor", "model analyis"])

# Set the title of the app
st.title("News Classifier")

# Add a description
st.write("Analyzing news articles")

# Text input for news articles
text = st.text_area("Enter Text", "")

# Button to classify the news article
if st.button("Classify"):
    # This is where the ML model prediction would happen
    # For now, we'll just display the entered text
    # You would replace this with your model's prediction logic
    st.write("Prediction with ML Models")
    st.write(f"Entered Text: {text}")

# Add a deploy button (for demonstration purposes)
st.button("Deploy")
