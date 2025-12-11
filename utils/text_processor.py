import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download NLTK resources (you might want to handle this in setup)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.7,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.fit_transform(processed_texts)
    
    def transform(self, texts):
        """
        Transform new texts using fitted vectorizer
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer to a file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self, filepath):
        """
        Load a fitted vectorizer from a file
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)

# Example usage
if __name__ == "__main__":
    # Sample data
    texts = [
        "This is a sample text for testing.",
        "Another example text for the vectorizer.",
        "Machine learning is fascinating!"
    ]
    
    # Initialize processor
    processor = TextProcessor()
    
    # Fit and transform
    X = processor.fit_transform(texts)
    
    print(f"Shape of TF-IDF matrix: {X.shape}")
    print(f"Vocabulary size: {len(processor.vectorizer.vocabulary_)}")
    
    # Save the vectorizer (for later use)
    processor.save_vectorizer("tfidf_vectorizer.pkl")