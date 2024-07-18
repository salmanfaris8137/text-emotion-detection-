import re 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import streamlit as st
import pandas as pd

# Streamlit app title
st.title("Text Emotion Prediction")

# Function to read data from a file
def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

# Function to create n-grams
def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

# Function to create features from text
def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

# Function to convert label
def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    return label.strip()

# List of emotions
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

# Load data
try:
    file = 'text.txt'
    data = read_data(file)
    st.success(f"Data loaded successfully. Number of instances: {len(data)}")
except FileNotFoundError:
    st.error("Error: 'text.txt' file not found. Please make sure the file exists in the same directory as this script.")
    st.stop()

# Prepare features and labels
X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

# Vectorize features
vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
clf = LinearSVC(random_state=123)
clf.fit(X_train, y_train)

# Evaluate the model
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

st.write(f"Model Training Accuracy: {train_acc:.2f}")
st.write(f"Model Test Accuracy: {test_acc:.2f}")

# Count label frequencies
label_freq = Counter(y_all)

# Display label frequencies
st.subheader("Emotion Distribution in Dataset")
df_freq = pd.DataFrame.from_dict(label_freq, orient='index', columns=['Count'])
df_freq = df_freq.sort_values('Count', ascending=False)
st.bar_chart(df_freq)

# Emoji dictionary
emoji_dict = {
    "joy": "😂", "fear": "😱", "anger": "😠", "sadness": "😢",
    "disgust": "😒", "shame": "😳", "guilt": "😳"
}

# User input for prediction
user_input = st.text_input("Enter any text for emotion prediction:")

if user_input:
    # Predict emotion
    features = create_feature(user_input, nrange=(1, 4))
    features = vectorizer.transform([features])
    prediction = clf.predict(features)[0]
    
    # Display result
    st.subheader("Predicted Emotion:")
    st.write(f"{prediction} {emoji_dict[prediction]}")

# Explanation of the model
st.subheader("How it works")
st.write("""
This emotion detection model uses the following steps:
1. Text preprocessing and feature extraction using n-grams
2. Training a Linear Support Vector Classifier (LinearSVC)
3. Predicting the emotion of new text inputs

The model is trained on a dataset of labeled emotions and can predict 
seven different emotions: joy, fear, anger, sadness, disgust, shame, and guilt.
""")