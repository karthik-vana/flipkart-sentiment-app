import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import joblib

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def load_data(file_path):
    """Load dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_text(text):
    """
    Clean text data:
    - Lowercase
    - Remove HTML tags
    - Remove special characters / numbers
    - Remove stopwords
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize (split by space)
    tokens = text.split()
    
    # Remove stopwords and Lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

def preprocess_pipeline(input_path, output_path):
    """
    Full preprocessing pipeline.
    """
    print("Loading data...")
    df = load_data(input_path)
    
    # Display shape and info
    print(f"Original Shape: {df.shape}")
    print(df.columns)
    
    # Print columns for verification
    print(f"Columns found: {df.columns.tolist()}")
    
    # Identify rating column
    if 'Ratings' in df.columns:
        rating_col = 'Ratings'
    elif 'Reviewer Rating' in df.columns:
        rating_col = 'Reviewer Rating'
    elif 'reviewer_rating' in df.columns:
        rating_col = 'reviewer_rating'
    elif 'rating' in df.columns:
        rating_col = 'rating'
    else:
        raise ValueError(f"Rating column not found. Available columns: {df.columns.tolist()}")

    # Identify title and text columns
    title_col = None
    if 'Review Title' in df.columns:
        title_col = 'Review Title'
    elif 'review_title' in df.columns:
        title_col = 'review_title'
    elif 'Review title' in df.columns:
        title_col = 'Review title'
        
    text_col = None
    if 'Review Text' in df.columns:
        text_col = 'Review Text'
    elif 'review_text' in df.columns:
        text_col = 'review_text'
    elif 'Review text' in df.columns:
        text_col = 'Review text'
        
    if not title_col or not text_col:
         print(f"Warning: Title or Text column not found perfectly. Using available text columns.")
         if text_col:
             df['review_full'] = df[text_col].fillna('')
         else:
             raise ValueError("No review text column found.")
    else:
        # Combine Title and Text
        print("Combining Title and Text...")
        df['review_full'] = df[title_col].fillna('') + " " + df[text_col].fillna('')
    
    # Clean Text
    print("Cleaning text...")
    df['cleaned_text'] = df['review_full'].apply(clean_text)

    # Create Target Variable (Sentiment)
    print("Creating sentiment target...")
    # Convert rating to numeric if needed
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    # Rating > 3 is Positive (1), else Negative (0)
    df['sentiment'] = df[rating_col].apply(lambda x: 1 if x > 3 else 0)

    
    # Handle missing values (empty strings after cleaning)
    df = df[df['cleaned_text'] != ""]
    
    print(f"Final Shape: {df.shape}")
    print("Saving cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    INPUT_FILE = os.path.join("data", "reviews_badminton", "data.csv")
    OUTPUT_FILE = os.path.join("data", "cleaned_data.csv")
    
    preprocess_pipeline(INPUT_FILE, OUTPUT_FILE)
