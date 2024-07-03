predict the emotional content of text input. 

Purpose: The project creates a machine learning model to classify text into different emotional categories.
Emotions detected: The system can identify seven emotions: joy, fear, anger, sadness, disgust, shame, and guilt.
Data processing:

Reads data from a text file ('text.txt')
Converts text to features using n-grams (1 to 4-grams)
Includes punctuation as features


Machine learning:

Uses scikit-learn for machine learning tasks
Implements a Linear Support Vector Classification (LinearSVC) model
Splits data into training and testing sets
Evaluates model performance using accuracy scores


User interface:

Utilizes Streamlit to create a simple web interface
Allows users to input text for emotion prediction


Output:

Predicts the emotion of the input text
Displays the result using an emoji corresponding to the predicted emotion


Additional analysis:

Includes code to analyze the frequency of different emotion labels in the dataset



This project combines natural language processing techniques with machine learning to create a practical application for detecting emotions in text, which could be useful in various fields such as sentiment analysis, customer service, or social media monitoring
