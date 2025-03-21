import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import os
from models.machine_learning import highlight_sentence_html, calculate_score_lr, clear_content
from models.naive import calculate_score_nb, NB_MODEL_PATH, NB_VECTORIZER_PATH

# File paths
MODEL_PATH = os.path.join('models', 'lr_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load logistic regression model
with open(MODEL_PATH, 'rb') as f:
    lr_model = pickle.load(f)

# Load TF-IDF vectorizer
with open(VECTORIZER_PATH, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load Naive Bayes model
with open(NB_MODEL_PATH, 'rb') as f:
    nb_model = pickle.load(f)

# Load Naive Bayes vectorizer
with open(NB_VECTORIZER_PATH, 'rb') as f:
    nb_vectorizer = pickle.load(f)


# Streamlit app title
st.title("ReviewAssist: Google Play Store Review Analyzer")

# Input for Google Play Store app URL or manual review entry
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Google Play Store App URL", "Manual Review Input"])

if input_mode == "Google Play Store App URL":
    # Input for Google Play Store app URL
    app_url = st.text_input("Enter the Google Play Store app URL:")
    
    if app_url:
        # Extract app ID from the URL (assuming a function exists for this)
        def extract_app_id(url):
            import re
            match = re.search(r'id=([^&]+)', url)
            return match.group(1) if match else None

        app_id = extract_app_id(app_url)
        
        if not app_id:
            st.error("Invalid Google Play Store URL.")
        else:
            st.write(f"Fetching reviews for App ID: {app_id}")
            
            # Fetch reviews (assuming a function exists for this)
            from google_play_scraper import reviews, Sort
            
            def get_app_reviews(app_id, count=100):
                try:
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

            reviews_df = get_app_reviews(app_id)
            
            if not reviews_df.empty:
                st.write(f"Fetched {len(reviews_df)} reviews.")
                st.dataframe(reviews_df.head())

                # Highlight the first review as an example
                first_review = reviews_df['content'].iloc[0]
                scores_lr = calculate_score_lr(first_review, lr_model, tfidf_vectorizer)
                scores_nb = calculate_score_nb(first_review, nb_model, nb_vectorizer)
                highlighted_html_lr = highlight_sentence_html(first_review, lr_model, tfidf_vectorizer)

                # Display highlighted sentence
                st.subheader("Highlighted Sentence (Logistic Regression)")
                components.html(highlighted_html_lr, height=300, scrolling=True)

                # Display sentiment scores comparison
                st.subheader("Sentiment Scores Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Logistic Regression:")
                    st.write(f"Positive: {scores_lr['positive']:.2f}")
                    st.write(f"Negative: {scores_lr['negative']:.2f}")
                with col2:
                    st.write("Naive Bayes:")
                    st.write(f"Positive: {scores_nb['positive']:.2f}")
                    st.write(f"Negative: {scores_nb['negative']:.2f}")
            else:
                st.warning("No reviews found.")
else:
    # Manual review input mode
    new_review = st.text_area("Enter a review:", "This app is terrible and crashes constantly.")

    if new_review:
        # Preprocess and analyze the review
        scores_lr = calculate_score_lr(new_review, lr_model, tfidf_vectorizer)
        scores_nb = calculate_score_nb(new_review, nb_model, nb_vectorizer)
        highlighted_html_lr = highlight_sentence_html(new_review, lr_model, tfidf_vectorizer)

        # Display highlighted sentence
        st.subheader("Highlighted Sentence (Logistic Regression)")
        components.html(highlighted_html_lr, height=300, scrolling=True)

        # Display sentiment scores comparison
        st.subheader("Sentiment Scores Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Logistic Regression:")
            st.write(f"Positive: {scores_lr['positive']:.2f}")
            st.write(f"Negative: {scores_lr['negative']:.2f}")

# Footer information
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Your Name](https://github.com/your-profile)")
