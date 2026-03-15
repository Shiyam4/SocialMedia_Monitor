import sys
import os

# Add project root folder to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from preprocessing import clean_text

# Load dataset
dataset_path = os.path.join(BASE_DIR, "dataset", "data.csv")
df = pd.read_csv(dataset_path)

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=200))
])

model.fit(X, y)

save_path = os.path.join(BASE_DIR, "model", "misinformation_model.pkl")
joblib.dump(model, save_path)

print("Model trained and saved successfully!")
