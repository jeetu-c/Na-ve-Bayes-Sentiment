import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sentiment Core", layout="wide")
st.title("🎬 IMDB Sentiment Processing Unit")

uploaded_csv = st.file_uploader("Drop IMDB Dataset Here", type="csv")
if uploaded_csv:
    with st.status("Training Bayes Model..."):
        df = pd.read_csv(uploaded_csv)
        df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
        X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X_tr = vectorizer.fit_transform(X_train)
        X_ts = vectorizer.transform(X_test)
        
        nb_model = MultinomialNB()
        nb_model.fit(X_tr, y_train)
        accuracy = accuracy_score(y_test, nb_model.predict(X_ts))
    
    st.balloons()
    st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
