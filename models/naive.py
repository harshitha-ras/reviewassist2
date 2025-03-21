from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from models.machine_learning import load_data, clear_content, train_model
import os
import pickle

# File paths for Naive Bayes model and vectorizer
NB_MODEL_PATH = os.path.join('models', 'nb_model.pkl')
NB_VECTORIZER_PATH = os.path.join('models', 'nb_vectorizer.pkl')

def train_naive_bayes():
    """Train Naive Bayes model on CountVectorizer features."""
    # Load and preprocess data
    df = load_data()

    # Convert text to numerical features using CountVectorizer
    nb_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 1), stop_words='english')
    X = nb_vectorizer.fit_transform(df['content'])
    y = df['sentiment']

    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X, y)

    # Save trained model and vectorizer to files
    with open(NB_MODEL_PATH, 'wb') as f:
        pickle.dump(nb_model, f)

    with open(NB_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(nb_vectorizer, f)

def calculate_score_nb(review: str, nb_model, nb_vectorizer):
    """
    Calculate sentiment scores for a review using Naive Bayes and CountVectorizer.
    """
    # Preprocess the review
    processed_review = clear_content(review)
    
    # Transform the review using CountVectorizer
    transformed_review = nb_vectorizer.transform([processed_review])
    
    # Predict probabilities for positive and negative sentiment
    probabilities = nb_model.predict_proba(transformed_review)[0]
    
    # Return sentiment probabilities (positive and negative)
    return {
        'positive': probabilities[nb_model.classes_.tolist().index('positive')],
        'negative': probabilities[nb_model.classes_.tolist().index('negative')]
    }

if __name__ == '__main__':
    train_model()
    train_naive_bayes()
    print("Model training completed.")
