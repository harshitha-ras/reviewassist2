import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import os
from models.machine_learning import (
    highlight_sentence_html, 
    calculate_score_lr,
    highlight_sentence_nb_html,  # New function for Naive Bayes
    calculate_score_nb,
    extract_keywords_rake           # New function for Naive Bayes
)

# File paths
MODEL_PATH = os.path.join('models', 'lr_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
NB_MODEL_PATH = os.path.join('models', 'nb_model.pkl')
NB_VECTORIZER_PATH = os.path.join('models', 'count_vectorizer.pkl')

# Load logistic regression model
with open(MODEL_PATH, 'rb') as f:
    lr_model = pickle.load(f)

# Load TF-IDF vectorizer
with open(VECTORIZER_PATH, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load Naive Bayes model
with open(NB_MODEL_PATH, 'rb') as f:
    nb_model = pickle.load(f)

# Load CountVectorizer for Naive Bayes
with open(NB_VECTORIZER_PATH, 'rb') as f:
    count_vectorizer = pickle.load(f)

# Verify loaded objects
print(type(lr_model))  # Should print <class 'sklearn.linear_model._logistic.LogisticRegression'>
print(type(tfidf_vectorizer))  # Should print <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
print(type(nb_model))  # Should print <class 'sklearn.naive_bayes.MultinomialNB'>
print(type(count_vectorizer))  # Should print <class 'sklearn.feature_extraction.text.CountVectorizer'>

# Streamlit app title
st.title("ReviewAssist: Google Play Store Review Analyzer")

# Sidebar for input options and model explanation
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Google Play Store App URL", "Manual Review Input"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Differences:**
- 🤖 **Logistic Regression**: 
  - Uses TF-IDF weighting
  - Considers feature importance
  - Better with imbalanced data
  
- 🧠 **Naive Bayes**:
  - Uses word counts
  - Assumes feature independence
  - Faster training time
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Your Name](https://github.com/your-profile)")

# Function to extract app ID from URL
def extract_app_id(url):
    import re
    match = re.search(r'id=([^&]+)', url)
    return match.group(1) if match else None

# Function to fetch app reviews
def get_app_reviews(app_id, count=100):
    try:
        from google_play_scraper import reviews, Sort
        result, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=count
        )
        reviews_data = [{'content': r.get('content', ''), 'score': r.get('score', None)} for r in result]
        return pd.DataFrame(reviews_data)
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return pd.DataFrame(columns=['content', 'score'])

# Input handling and analysis
if input_mode == "Google Play Store App URL":
    # Input for Google Play Store app URL
    app_url = st.text_input("Enter the Google Play Store app URL:")
    
    if app_url:
        app_id = extract_app_id(app_url)
        
        if not app_id:
            st.error("Invalid Google Play Store URL.")
        else:
            st.write(f"Fetching reviews for App ID: {app_id}")
            reviews_df = get_app_reviews(app_id)
            
            if not reviews_df.empty:
                st.write(f"Fetched {len(reviews_df)} reviews.")
                st.dataframe(reviews_df.head())

                # Highlight the first review as an example
                first_review = reviews_df['content'].iloc[0]

                # Create three columns for comparison
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Naive Bayes Analysis")
                    scores_nb = calculate_score_nb(first_review, nb_model, count_vectorizer)
                    highlighted_nb = f'<div style="background-color: white; padding: 10px; border-radius: 5px;">{highlight_sentence_nb_html(first_review, nb_model, count_vectorizer)}</div>'
                    components.html(highlighted_nb, height=300, scrolling=True)
                    st.write(f"Positive: {scores_nb['positive']:.2%}")
                    st.write(f"Negative: {scores_nb['negative']:.2%}")

                with col2:
                    st.subheader("Logistic Regression Analysis")
                    scores_lr = calculate_score_lr(first_review, lr_model, tfidf_vectorizer)
                    highlighted_lr = f'<div style="background-color: white; padding: 10px; border-radius: 5px;">{highlight_sentence_html(first_review, lr_model, tfidf_vectorizer)}</div>'
                    components.html(highlighted_lr, height=300, scrolling=True)
                    st.write(f"Positive: {scores_lr['positive']:.2%}")
                    st.write(f"Negative: {scores_lr['negative']:.2%}")

                with col3:
                    st.subheader("RAKE Keyword Extraction")
                    keywords = extract_keywords_rake(first_review)
                    st.write("Top 5 Keywords:")
                    for keyword in keywords:
                        st.write(f"- {keyword}")

    else:
        # Manual review input mode
        new_review = st.text_area("Enter a review:", "This app is terrible and crashes constantly.")

        if new_review:
            # Create three columns for comparison
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Naive Bayes Analysis")
                scores_nb = calculate_score_nb(new_review, nb_model, count_vectorizer)
                highlighted_nb = f'<div style="background-color: white; padding: 10px; border-radius: 5px;">{highlight_sentence_nb_html(first_review, nb_model, count_vectorizer)}</div>'
                components.html(highlighted_nb, height=300, scrolling=True)
                st.write(f"Positive: {scores_nb['positive']:.2%}")
                st.write(f"Negative: {scores_nb['negative']:.2%}")

            with col2:
                st.subheader("Logistic Regression Analysis")
                scores_lr = calculate_score_lr(new_review, lr_model, tfidf_vectorizer)
                highlighted_lr = f'<div style="background-color: white; padding: 10px; border-radius: 5px;">{highlight_sentence_html(first_review, lr_model, tfidf_vectorizer)}</div>'
                components.html(highlighted_lr, height=300, scrolling=True)
                st.write(f"Positive: {scores_lr['positive']:.2%}")
                st.write(f"Negative: {scores_lr['negative']:.2%}")

            with col3:
                st.subheader("RAKE Keyword Extraction")
                keywords = extract_keywords_rake(new_review)
                st.write("Top 5 Keywords:")
                for keyword in keywords:
                    st.write(f"- {keyword}")