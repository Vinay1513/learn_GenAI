# spam_classifier.py

import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load dataset
df = pd.read_csv("../data/spam.csv", encoding="latin-1", usecols=[0, 1])
df.columns = ['label', 'message']
print(df.head())
# filepath: c:\Users\Vinay\Desktop\ML\supervised_learning\spam-classifier\src\main.py# filepath: c:\Users\Vinay\Desktop\ML\supervised_learning\spam-classifier\src\main.py# filepath: c:\Users\Vinay\Desktop\ML\supervised_learning\spam-classifier\src\main.py# filepath: c:\Users\Vinay\Desktop\ML\supervised_learning\spam-classifier\src\main.pydf.columns = ['label', 'message']


# Step 2: Text Preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned'] = df['message'].apply(preprocess)

# Step 3: Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
