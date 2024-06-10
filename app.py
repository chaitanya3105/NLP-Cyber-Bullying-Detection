import streamlit as st
import pickle
import pandas as pd

# Load the model
model_filename = 'cyberbullying_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Cyberbullying Detection')

# User input
text_input = st.text_area("Enter the text you want to analyze:", "")

# Feature extraction functions
def extract_features(text):
    # Dummy functions for example, replace with actual functions
    # that were used during the model training
    num_noun_phrases = len([word for word in text.split() if word.lower() in ['noun1', 'noun2']])
    num_verbs = len([word for word in text.split() if word.lower() in ['verb1', 'verb2']])
    return {'Cleaned_Text': text, 'num_noun_phrases': num_noun_phrases, 'num_verbs': num_verbs}

if st.button('Predict'):
    if text_input:
        features = extract_features(text_input)
        features_df = pd.DataFrame([features])
        
        prediction = model.predict(features_df)[0]
        prediction_prob = model.predict_proba(features_df)[0]

        st.write(f"Prediction: {'Cyberbullying' if prediction == 1 else 'Not Cyberbullying'}")
        st.write(f"Probability: {prediction_prob[prediction]:.2f}")
    else:
        st.write("Please enter some text to analyze.")
