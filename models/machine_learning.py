import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])

import os
import pandas as pd

# Define the path to the dataset relative to the project root
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sentiment-analysis-dataset-google-play-app-reviews.csv')

# Verify that the file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"The dataset file was not found at {DATA_PATH}. Please check the file path.")

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Print the first few rows of the dataset for verification
print("Dataset loaded successfully!")
print(df.head())


MODEL_PATH = os.path.join('lr_model.pkl')
VECTORIZER_PATH = os.path.join('tfidf_vectorizer.pkl')


# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Expand contractions (e.g., "can't" -> "cannot")
    text = contractions.fix(text)
    
    # Convert to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize and remove stopwords, then lemmatize
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(processed_tokens)

def load_data():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # Add sentiment column based on score
    df = add_sentiment_column(df)
    
    # Preprocess the content column
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    return df


def add_sentiment_column(df):
    # Map scores to sentiment labels
    def map_sentiment(score):
        if score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment'] = df['score'].apply(map_sentiment)
    return df


# Train model
def train_model():
    # Load and preprocess data
    df = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Convert text data into TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)

    # Evaluate model performance
    y_pred = lr_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model and vectorizer to files
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(lr_model, f)

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)


if __name__ == '__main__':
    train_model()
