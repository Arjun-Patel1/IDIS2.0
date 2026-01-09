import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "C:/Users/arjun/Downloads/IDIS2.0/data/tickets.csv"
MODEL_DIR = "C:/Users/arjun/Downloads/IDIS2.0/models/"

df = pd.read_csv(DATA_PATH)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(df["text"], df["label"])

with open(MODEL_DIR + "classifier.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["clf"], f)

with open(MODEL_DIR + "vectorizer.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps["tfidf"], f)

print("✅ Classifier trained and saved")
