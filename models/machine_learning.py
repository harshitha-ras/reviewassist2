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

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])
nltk.download('punkt', force=True)
# Set a custom NLTK data path (e.g., within your project directory)
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download necessary resources
nltk.download('punkt', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

# File paths
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sentiment-analysis-dataset-google-play-app-reviews.csv')

# Verify that the file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"The dataset file was not found at {DATA_PATH}. Please check the file path.")

# Load the dataset
df = pd.read_csv(DATA_PATH)
MODEL_PATH = os.path.join('models', 'lr_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Text preprocessing function
def clear_content(content):
    """Preprocess text by expanding contractions, lemmatizing, and removing special characters."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Step 1: Expand contractions (e.g., "can't" -> "cannot")
    content = contractions.fix(content)
    
    # Step 2: Convert to lowercase and remove non-alphabetic characters
    content = content.lower()
    content = re.sub(r'[^a-z\s]', '', content)
    
    # Step 3: Tokenize and remove stopwords, then lemmatize words
    tokens = word_tokenize(content)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(processed_tokens)

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



if __name__ == '__main__':
    train_model()
    print("Model training completed.")