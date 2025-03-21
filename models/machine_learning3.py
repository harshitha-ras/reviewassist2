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

# Highlight sentence based on sentiment scores
def highlight_sentence_html(sentence: str, sentiment_dict: dict, similarity_threshold=0.5):
    """Generate HTML with highlighted words based on sentiment scores."""
    words = sentence.split()
    highlighted_words = []

    for word in words:
        clean_word = word.lower().strip('.,!?;:()[]{}""\'')
        value = sentiment_dict.get(clean_word, 0)  # Get sentiment score for the word
        
        if value < 0:
            # Negative word -> red background color
            brightness = 255 - int(abs(value) * 255)
            color = f"rgb(255, {brightness}, {brightness})"
            highlighted_words.append(f'<span style="background-color: {color};">{word}</span>')
        elif value > 0:
            # Positive word -> green background color
            brightness = 255 - int(value * 255)
            color = f"rgb({brightness}, 255, {brightness})"
            highlighted_words.append(f'<span style="background-color: {color};">{word}</span>')
        else:
            highlighted_words.append(word)  # Neutral or unmatched word
    
    return ' '.join(highlighted_words)

import numpy as np

def score_to_trans(x):
    """Transformation function (2sigmoid(x)-1) that maps values to a number between +1 and -1."""
    return 2 * 1 / (1 + np.exp(-x)) - 1

def calculate_score_lr(new_input: str, positive_words: dict, negative_words: dict, similarity_threshold=0.5):
    """Assign an adjusted weight of each fragment in new input."""
    result = {}
    new_review_cleared = clear_content(new_input)
    new_review_list = new_review_cleared.split(" ")
    agg_words = positive_words | negative_words  # Combine positive and negative words
    
    for frag in new_review_list:
        acc_score = 0
        matches = 0
        for k, v in agg_words.items():
            try:
                similarity = word2vec_model.similarity(frag, k)
                if similarity >= similarity_threshold:
                    acc_score += similarity * v  # Generate new score for color
                    matches += 1
            except KeyError:
                pass  # Skip fragments not in the model vocabulary
        
        result[frag] = 0 if matches == 0 else acc_score / matches
    
    # Generate transparency based on normal CDF
    for w in result:
        if result[w] != 0:  # Faster processing
            result[w] = score_to_trans(result[w])
    
    return result


# Example usage (only when running this script directly)
if __name__ == '__main__':
    train_model()
