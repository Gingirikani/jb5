import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st

# Load dataset from CSV files
train_df = pd.read_csv('train.csv')  # Ensure the CSV files are in the correct format
test_df = pd.read_csv('test.csv')

# Split dataset into features and labels
X_train = train_df['text'].values
y_train = train_df['label'].values
X_test = test_df['text'].values
y_test = test_df['label'].values

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train_tfidf, y_train)

# Save models
with open('model_lr.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
with open('model_nb.pkl', 'wb') as f:
    pickle.dump(nb, f)
with open('model_svm.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Streamlit App
st.title("News Classifier")
st.write("Classify news articles using different models")

# Load models and vectorizer
models = {
    "Logistic Regression": pickle.load(open('model_lr.pkl', 'rb')),
    "Naive Bayes": pickle.load(open('model_nb.pkl', 'rb')),
    "Support Vector Machine": pickle.load(open('model_svm.pkl', 'rb'))
}
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

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


