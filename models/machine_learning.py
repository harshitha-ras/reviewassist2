import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from rake_nltk import Rake
from nltk.corpus import WordListCorpusReader as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append('nltk_data/tokenizers/punkt/english.pickle')

# File paths
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sentiment-analysis-dataset-google-play-app-reviews.csv')

# Verify that the file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"The dataset file was not found at {DATA_PATH}. Please check the file path.")

# Load the dataset
df = pd.read_csv(DATA_PATH)
MODEL_PATH = os.path.join('models', 'lr_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
NB_MODEL_PATH = os.path.join('models', 'nb_model.pkl')
NB_VECTORIZER_PATH = os.path.join('models', 'count_vectorizer.pkl')

def clear_content(content):
    """Preprocess text by expanding contractions, lemmatizing, and removing stopwords."""
    
    # Download required NLTK resources if not already downloaded
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'{resource}')
        except LookupError:
            nltk.download(resource)
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Expand contractions
    content = contractions.fix(content)
    
    # Tokenize the content
    tokens = word_tokenize(content)
    
    # Remove stopwords and lemmatize tokens
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    
    # Join the tokens back into a string
    processed = ' '.join(filtered_tokens)
    
    return processed

# Load dataset and preprocess it
def load_data():
    """Load dataset, preprocess text, and create sentiment labels."""
    df = pd.read_csv(DATA_PATH)
    
    # Select relevant columns and drop missing values
    df = df[['content', 'score']].dropna()
    
    # Create sentiment labels based on score values
    df['sentiment'] = df['score'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
    
    # Exclude neutral reviews for binary classification
    df = df[df['sentiment'] != 'neutral']
    
    # Preprocess the content column for text analysis
    df['content'] = df['content'].apply(clear_content)
    
    return df

# Train logistic regression model
def train_model():
    """Train logistic regression model on TF-IDF features."""
    # Load and preprocess data
    df = load_data()

    # Extract features (TF-IDF) and labels (sentiment)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['content'])
    y = df['sentiment']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # Train logistic regression model with balanced class weights
    lr_model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=0)
    lr_model.fit(X_train, y_train)

    # Evaluate model performance on test data
    y_pred = lr_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save trained model and vectorizer to files for later use in Streamlit app
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(lr_model, f)

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

def extract_keywords_rake(text, num_keywords=5):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:num_keywords]

# Highlight sentence based on sentiment scores from logistic regression coefficients
def highlight_sentence_html(sentence: str, lr_model, tfidf_vectorizer):
    """
    Generate HTML with highlighted words based on logistic regression predictions.
    """
    words = sentence.split()
    highlighted_words = []

    for word in words:
        # Preprocess each word
        processed_word = clear_content(word)
        
        # Transform the word using TF-IDF vectorizer
        transformed_word = tfidf_vectorizer.transform([processed_word])
        
        # Predict probabilities for positive and negative sentiment
        probabilities = lr_model.predict_proba(transformed_word)[0]
        
        # Highlight word based on sentiment probabilities
        positive_prob = probabilities[lr_model.classes_.tolist().index('positive')]
        negative_prob = probabilities[lr_model.classes_.tolist().index('negative')]

        if positive_prob > negative_prob:
            # Positive word -> green background color
            brightness = int(255 * positive_prob)
            color = f"rgb({brightness}, 255, {brightness})"
            highlighted_words.append(f'<span style="background-color: {color};">{word}</span>')
        elif negative_prob > positive_prob:
            # Negative word -> red background color
            brightness = int(255 * negative_prob)
            color = f"rgb(255, {brightness}, {brightness})"
            highlighted_words.append(f'<span style="background-color: {color};">{word}</span>')
        else:
            # Neutral word -> no highlight
            highlighted_words.append(word)

    return ' '.join(highlighted_words)

     
def calculate_score_lr(review: str, lr_model, tfidf_vectorizer):
    """
    Calculate sentiment scores for a review using logistic regression and TF-IDF.
    """
    # Preprocess the review
    processed_review = clear_content(review)
    
    # Transform the review using TF-IDF vectorizer
    transformed_review = tfidf_vectorizer.transform([processed_review])
    
    # Predict probabilities for positive and negative sentiment
    probabilities = lr_model.predict_proba(transformed_review)[0]
    
    # Return sentiment probabilities (positive and negative)
    return {
        'positive': probabilities[lr_model.classes_.tolist().index('positive')],
        'negative': probabilities[lr_model.classes_.tolist().index('negative')]
    }

NB_MODEL_PATH = os.path.join('models', 'nb_model.pkl')
NB_VECTORIZER_PATH = os.path.join('models', 'count_vectorizer.pkl')

def highlight_sentence_nb_html(sentence: str, nb_model, count_vectorizer):
    """Naive Bayes version of highlighting"""
    words = sentence.split()
    vocab = count_vectorizer.vocabulary_
    
    highlighted_words = []
    for word in words:
        clean_word = clear_content(word)
        if clean_word in vocab:
            # Get log probabilities for each class
            word_idx = vocab[clean_word]
            log_prob_pos = nb_model.feature_log_prob_[0][word_idx]
            log_prob_neg = nb_model.feature_log_prob_[1][word_idx]
            
            # Calculate intensity based on probability difference
            intensity = abs(log_prob_pos - log_prob_neg)
            if log_prob_pos > log_prob_neg:
                color = f"hsl(120, 100%, {100 - (intensity*50)}%)"
            else:
                color = f"hsl(0, 100%, {100 - (intensity*50)}%)"
                
            highlighted_words.append(f'<span style="background-color: {color}">{word}</span>')
        else:
            highlighted_words.append(word)
            
    return ' '.join(highlighted_words)

def calculate_score_nb(review: str, nb_model, count_vectorizer):
    """Naive Bayes scoring"""
    processed = clear_content(review)
    transformed = count_vectorizer.transform([processed])
    probs = nb_model.predict_proba(transformed)[0]
    return {
        'positive': probs[nb_model.classes_.tolist().index('positive')],
        'negative': probs[nb_model.classes_.tolist().index('negative')]
    }


def train_naive_bayes():
    """Train and save Naive Bayes model"""
    df = load_data()
    
    # CountVectorizer instead of TF-IDF
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(df['content'])
    y = df['sentiment']

    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X, y)

    # Save model and vectorizer
    with open(NB_MODEL_PATH, 'wb') as f:
        pickle.dump(nb_model, f)
        
    with open(NB_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(count_vectorizer, f)

if __name__ == '__main__':
    wordnet.ensure_loaded()
    train_model()
    train_naive_bayes()
    print("Model training completed.")