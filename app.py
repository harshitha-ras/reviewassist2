import streamlit as st
import pandas as pd
import re
from google_play_scraper import app
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

nltk.download(['punkt', 'wordnet', 'stopwords'])

with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def clear_content(content):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    content = contractions.fix(content)
    content = content.lower()
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    tokens = word_tokenize(content)
    
    cleared = []
    for word in tokens:
        if (word not in stop_words) and len(word) > 2:
            cleared.append(lemmatizer.lemmatize(word))
    return ' '.join(cleared)

def extract_app_id(url):
    match = re.search(r'id=([^&]+)', url)
    return match.group(1) if match else None

def get_app_reviews(app_id, count=100):
    result = app(
        app_id,
        lang='en',
        country='us'
    )
    reviews = result['reviews']
    return pd.DataFrame(reviews, columns=['content', 'score'])

def predict_sentiment_and_topic(review):
    cleaned_review = clear_content(review)
    vectorized_review = tfidf.transform([cleaned_review])
    sentiment = lr_model.predict(vectorized_review)[0]
    
    word_importance = vectorized_review.toarray()[0] * tfidf.idf_
    most_important_word_index = word_importance.argmax()
    topic = tfidf.get_feature_names_out()[most_important_word_index]
    
    return sentiment, topic

st.title("Google Play Store App Review Analyzer")

app_url = st.text_input("Enter the Google Play Store app URL:")

if app_url:
    app_id = extract_app_id(app_url)
    if app_id:
        st.write(f"Analyzing reviews for app ID: {app_id}")
        
        reviews_df = get_app_reviews(app_id)
        
        for _, review in reviews_df.iterrows():
            sentiment, topic = predict_sentiment_and_topic(review['content'])
            
            st.write("---")
            st.write(f"Review: {review['content']}")
            st.write(f"Topic: **{topic}**")
            st.write(f"Sentiment: **{sentiment}**")
    else:
        st.error("Invalid Google Play Store URL. Please enter a valid URL.")
