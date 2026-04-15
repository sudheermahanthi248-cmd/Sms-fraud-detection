from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import nltk
import re
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__, static_folder='.')

# ─────────────────────────────────────────────
# TRAIN MODEL ON STARTUP
# ─────────────────────────────────────────────
print("Setting up model...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

model = None
tfidf = None
model_accuracy = 0

def train_model():
    global model, tfidf, model_accuracy

    if not os.path.exists('data/spam.csv'):
        print("WARNING: data/spam.csv not found. Using demo mode.")
        return False

    df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['cleaned']   = df['message'].apply(preprocess)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label_enc'],
        test_size=0.2, random_state=42, stratify=df['label_enc']
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    model = LinearSVC(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test_tfidf)
    model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"Model trained! Accuracy: {model_accuracy}%")
    return True

trained = train_model()

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Demo mode if model not trained
    if model is None or tfidf is None:
        spam_keywords = ['free', 'win', 'winner', 'prize', 'urgent', 'click', 'claim',
                         'congratulations', 'selected', 'offer', 'cash', 'call now']
        is_spam = any(kw in message.lower() for kw in spam_keywords)
        label = 'SPAM' if is_spam else 'HAM'
        confidence = 87.5 if is_spam else 92.3
        return jsonify({
            'label': label,
            'confidence': confidence,
            'message': message,
            'demo_mode': True
        })

    cleaned    = preprocess(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Get confidence using decision function
    decision = model.decision_function(vectorized)[0]
    import numpy as np
    confidence = round(min(99.9, max(50.0, 50 + abs(float(decision)) * 25)), 1)

    label = 'SPAM' if prediction == 1 else 'HAM'
    return jsonify({
        'label': label,
        'confidence': confidence,
        'message': message,
        'demo_mode': False
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': model is not None,
        'accuracy': model_accuracy,
        'demo_mode': not trained
    })

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  SMS Spam Detection Web App")
    print("  Open: http://localhost:5000")
    print("="*45 + "\n")
    app.run(debug=True, port=5000)
