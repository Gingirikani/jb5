import streamlit as st

# Load trained models
import pickle

models = {
    "Logistic Regression": log_reg,
    "Naive Bayes": nb,
    "Support Vector Machine": svc
}

# Load models and vectorizer
models = {
    "Logistic Regression": pickle.load(open('model_lr.pkl', 'rb')),
    "Naive Bayes": pickle.load(open('model_nb.pkl', 'rb')),
    "Support Vector Machine": pickle.load(open('svc_model.pkl', 'rb'))
}

count_vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

# Streamlit App
st.title("News Classifier")
st.write("Classify news articles using different models")

# Input text
input_text = st.text_area("Enter News Article")

# Model selection
model_choice = st.selectbox("Choose Model", list(models.keys()))

# Prediction
if st.button("Classify"):
    model = models[model_choice]
    input_vectorized = vectorizer.transform([input_text])
    prediction = model.predict(input_vectorized)[0]
    prediction_proba = model.predict_proba(input_vectorized)[0]
    
    st.write(f"Predicted Category: {target_names[prediction]}")
    st.write(f"Prediction Confidence: {prediction_proba[prediction]:.2f}")


