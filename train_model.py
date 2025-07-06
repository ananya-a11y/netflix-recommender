import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Step 1: Load data
df = pd.read_csv('main_data.csv')  # Make sure this file exists
df = df.dropna(subset=['comb'])    # Drop rows where 'comb' column is null

# Step 2: Fit TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['comb'])

# Step 3: Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ vectorizer.pkl created successfully")

