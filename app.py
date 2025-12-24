import pandas as pd
import numpy as np
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("data/fake_job_postings.csv")

# Keep only needed columns
data = data[['description', 'fraudulent']]
data.dropna(inplace=True)

# ----------------------------
# 2. Text Preprocessing
# ----------------------------
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['description'] = data['description'].apply(clean_text)

# ----------------------------
# 3. Feature Extraction
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['description'])
y = data['fraudulent']

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Train Model
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# 6. Evaluation
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ----------------------------
# 7. Save Model
# ----------------------------
joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully!")

# ----------------------------
# 8. Prediction Function
# ----------------------------
def predict_job(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        return "🚨 FAKE JOB POSTING"
    else:
        return "✅ REAL JOB POSTING"


# ----------------------------
# 9. Test Prediction
# ----------------------------
sample_job = """
We are hiring data entry operators.
No experience needed.
Work from home.
Earn 3000 per day.
"""

print(predict_job(sample_job))
