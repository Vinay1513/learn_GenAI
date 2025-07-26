# Supervised Learning Projects Documentation

This document provides a comprehensive explanation of both machine learning projects in this workspace: **Spam Classifier** and **Student Performance Predictor**.

## 📁 Project Overview

### 1. Spam Classifier Project
**Location:** `spam-classifier/`
**Purpose:** Text classification to identify spam vs legitimate messages using Natural Language Processing (NLP) and Machine Learning.

### 2. Student Performance Predictor Project  
**Location:** `student_performance_predictor/`
**Purpose:** Binary classification to predict whether a student will pass or fail based on academic metrics using a web interface.

---

## 🔍 Spam Classifier - Detailed Analysis

### 📂 Project Structure
```
spam-classifier/
├── src/
│   └── main.py          # Main classification script
├── data/
│   └── spam.csv         # Training dataset
├── requirements.txt      # Dependencies
└── README.md           # Project documentation
```

### 📝 Code Analysis: `src/main.py`

#### **Imports and Dependencies:**

```python
import pandas as pd                    # Data manipulation and analysis
import string                          # String operations and punctuation handling
import matplotlib.pyplot as plt        # Data visualization (imported but not used)
from sklearn.model_selection import train_test_split          # Dataset splitting
from sklearn.feature_extraction.text import CountVectorizer   # Text to numerical conversion
from sklearn.naive_bayes import MultinomialNB                # Naive Bayes classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Model evaluation
import nltk                           # Natural Language Processing toolkit
from nltk.corpus import stopwords     # Common words to remove from text
```

#### **Why These Imports Are Used:**

1. **pandas (pd):** 
   - Used for reading CSV data (`pd.read_csv()`)
   - Data manipulation and preprocessing
   - DataFrame operations for text cleaning

2. **string:** 
   - Provides access to punctuation characters
   - Used in text preprocessing to remove punctuation marks

3. **sklearn.model_selection.train_test_split:**
   - Splits dataset into training (80%) and testing (20%) sets
   - Ensures model evaluation on unseen data

4. **sklearn.feature_extraction.text.CountVectorizer:**
   - Converts text data into numerical features (Bag of Words)
   - Creates a matrix where each row represents a message and each column represents a word

5. **sklearn.naive_bayes.MultinomialNB:**
   - Naive Bayes classifier suitable for text classification
   - Works well with count-based features
   - Fast training and prediction

6. **sklearn.metrics:**
   - `accuracy_score`: Measures overall prediction accuracy
   - `confusion_matrix`: Shows true vs predicted classifications
   - `classification_report`: Detailed metrics (precision, recall, F1-score)

7. **nltk and stopwords:**
   - Natural Language Processing toolkit
   - Removes common words (the, is, at, which, on) that don't add meaning

#### **Key Concepts and Workflow:**

1. **Data Loading:**
   ```python
   df = pd.read_csv("../data/spam.csv", encoding="latin-1", usecols=[0, 1])
   df.columns = ['label', 'message']
   ```
   - Reads CSV file with spam/ham labels and messages
   - Uses 'latin-1' encoding for special characters

2. **Text Preprocessing:**
   ```python
   def preprocess(text):
       text = text.lower()                                    # Convert to lowercase
       text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
       text = ' '.join(word for word in text.split() if word not in stop_words)    # Remove stopwords
       return text
   ```
   - **Lowercase conversion:** Standardizes text format
   - **Punctuation removal:** Eliminates special characters that don't add meaning
   - **Stopword removal:** Removes common words that don't help classification

3. **Feature Extraction:**
   ```python
   vectorizer = CountVectorizer()
   X = vectorizer.fit_transform(df['cleaned'])
   ```
   - Converts cleaned text into numerical features
   - Creates a sparse matrix where each word becomes a feature

4. **Model Training:**
   ```python
   model = MultinomialNB()
   model.fit(X_train, y_train)
   ```
   - Uses Naive Bayes algorithm for classification
   - Trains on 80% of the data

5. **Evaluation:**
   - Tests on 20% of unseen data
   - Provides accuracy, confusion matrix, and detailed classification report

---

## 🎓 Student Performance Predictor - Detailed Analysis

### 📂 Project Structure
```
student_performance_predictor/
├── app/
│   ├── app.py           # Streamlit web application
│   └── helper.py        # Helper functions
├── model/
│   └── student_model.pkl # Trained machine learning model
├── data/
│   └── student_data.csv  # Training dataset
└── notebooks/
    └── student_model_training.ipynb  # Model training notebook
```

### 📝 Code Analysis: `app/app.py`

#### **Imports and Dependencies:**

```python
import streamlit as st        # Web application framework
import pandas as pd          # Data manipulation
import joblib               # Model serialization/deserialization
import os                   # Operating system interface
```

#### **Why These Imports Are Used:**

1. **streamlit (st):**
   - Creates interactive web applications
   - Provides UI components (sliders, buttons, metrics)
   - Handles user input and displays results

2. **pandas (pd):**
   - Creates DataFrame for model input
   - Ensures proper data format for prediction

3. **joblib:**
   - Loads the pre-trained machine learning model
   - Efficient serialization for large objects

4. **os:**
   - Handles file paths across different operating systems
   - Constructs proper path to model file

#### **Key Concepts and Workflow:**

1. **Model Loading:**
   ```python
   model_path = os.path.join("..", "model", "student_model.pkl")
   model = joblib.load(model_path)
   ```
   - Loads the pre-trained Logistic Regression model
   - Uses relative path to access model file

2. **User Interface:**
   ```python
   study_hours = st.slider("Study Hours per Day", 0, 12, 6)
   attendance = st.slider("Attendance (%)", 0, 100, 75)
   assignment_score = st.slider("Assignment Score", 0, 100, 70)
   ```
   - Creates interactive sliders for user input
   - Sets reasonable default values

3. **Prediction Process:**
   ```python
   input_df = pd.DataFrame([[study_hours, attendance, assignment_score]],
                          columns=["StudyHours", "attendance", "PreviousScore"])
   prediction = model.predict(input_df)[0]
   ```
   - Converts user input to DataFrame format
   - Matches the format used during training
   - Makes binary prediction (0 = FAIL, 1 = PASS)

4. **Result Display:**
   - Shows input values in organized metrics
   - Displays prediction with color-coded success/error messages

### 📝 Code Analysis: `app/helper.py`

```python
def load_model():
    import joblib
    return joblib.load("../model/student_model.pkl")
```

**Purpose:** 
- Provides a helper function to load the model
- Separates model loading logic from main application
- Could be used for model reloading or testing

### 📝 Code Analysis: `notebooks/student_model_training.ipynb`

#### **Model Training Process:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
```

#### **Training Data:**
```python
data = pd.DataFrame({
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [40, 50, 60, 70, 80, 90, 95, 100],
    "PreviousScore": [30, 40, 50, 60, 70, 80, 90, 95],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = FAIL, 1 = PASS
})
```

**Features:**
- **StudyHours:** Hours spent studying per day (1-8 hours)
- **attendance:** Class attendance percentage (40-100%)
- **PreviousScore:** Previous assignment scores (30-95%)

**Target:**
- **Result:** Binary outcome (0 = FAIL, 1 = PASS)

#### **Model Training:**
```python
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "../model/student_model.pkl")
```

**Why Logistic Regression:**
- Suitable for binary classification problems
- Provides probability scores
- Interpretable coefficients
- Works well with small datasets

---

## 🔧 Technical Concepts Explained

### **Machine Learning Concepts:**

1. **Supervised Learning:**
   - Both projects use labeled data for training
   - Models learn patterns from input-output pairs

2. **Text Classification (Spam Classifier):**
   - **Bag of Words:** Represents text as word frequency vectors
   - **TF-IDF:** Could be used for better feature weighting
   - **Naive Bayes:** Assumes independence between features

3. **Binary Classification (Student Predictor):**
   - **Logistic Regression:** Predicts probability of pass/fail
   - **Decision Boundary:** Threshold-based classification

### **Data Preprocessing:**

1. **Text Preprocessing:**
   - Lowercase conversion
   - Punctuation removal
   - Stopword removal
   - Tokenization

2. **Feature Engineering:**
   - Numerical feature scaling (if needed)
   - Categorical encoding (if needed)

### **Model Evaluation:**

1. **Classification Metrics:**
   - **Accuracy:** Overall correct predictions
   - **Precision:** True positives / (True positives + False positives)
   - **Recall:** True positives / (True positives + False negatives)
   - **F1-Score:** Harmonic mean of precision and recall

2. **Confusion Matrix:**
   - Shows true vs predicted classifications
   - Helps identify model biases

---

## 🚀 How to Run the Projects

### **Spam Classifier:**
```bash
cd spam-classifier
pip install -r requirements.txt
python src/main.py
```

### **Student Performance Predictor:**
```bash
cd student_performance_predictor
pip install streamlit pandas scikit-learn joblib
streamlit run app/app.py
```

---

## 📊 Key Differences Between Projects

| Aspect | Spam Classifier | Student Performance Predictor |
|--------|----------------|------------------------------|
| **Data Type** | Text (NLP) | Numerical features |
| **Algorithm** | Naive Bayes | Logistic Regression |
| **Interface** | Command line | Web application (Streamlit) |
| **Features** | Text preprocessing | Direct numerical input |
| **Output** | Spam/Ham classification | Pass/Fail prediction |
| **Complexity** | Higher (NLP pipeline) | Simpler (direct features) |

---

## 🎯 Learning Outcomes

### **Spam Classifier Teaches:**
- Natural Language Processing (NLP)
- Text preprocessing techniques
- Feature extraction from text
- Naive Bayes classification
- Text classification pipeline

### **Student Performance Predictor Teaches:**
- Web application development
- Model deployment
- User interface design
- Binary classification
- Real-world ML application

Both projects demonstrate different aspects of supervised learning and provide hands-on experience with machine learning workflows from data preprocessing to model deployment. 