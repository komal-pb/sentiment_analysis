import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import pickle
from preprocess import TextPreprocessor


# Load dataset
df = pd.read_csv("airline_data/tweets.csv")
df['target'] = df['airline_sentiment'].map({'negative':0, 'neutral':1, 'positive':2})
df = df[['text', 'target']]

# Upsample minority classes
df_negative = df[df.target == 0]
df_neutral  = df[df.target == 1]
df_positive = df[df.target == 2]

df_neutral_upsampled  = resample(df_neutral, replace=True, n_samples=len(df_negative), random_state=42)
df_positive_upsampled = resample(df_positive, replace=True, n_samples=len(df_negative), random_state=42)

df_balanced = pd.concat([df_negative, df_neutral_upsampled, df_positive_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

X = df_balanced['text']
Y = df_balanced['target']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Pipeline
model = Pipeline([
    ('preprocess', TextPreprocessor()),
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
])

model.fit(X_train, Y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
