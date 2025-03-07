import streamlit as st
import pickle
import nltk
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the trained model and tokenizer
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

def check_spam(text):
    text = transform_text(text)
    sms_sequence = tokenizer.texts_to_sequences([text])
    sms_padded = pad_sequences(sms_sequence, maxlen=50)  # Adjust maxlen if necessary
    prediction = model.predict(sms_padded)
    prob = prediction[0][0]
    return {"spam_probability": prob, "not_spam_probability": 1.0 - prob}

# Streamlit UI
st.title("SMS Spam Detector")
st.write("Enter an SMS message below to check if it's spam.")

sms_text = st.text_area("Enter SMS text:")

if st.button("Check Spam"):
    if sms_text:
        result = check_spam(sms_text)
        st.write(f"Spam Probability: {result['spam_probability']:.2f}")
        st.write(f"Not Spam Probability: {result['not_spam_probability']:.2f}")
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.header("About")
st.sidebar.write("This app uses a machine learning model to predict whether an SMS message is spam or not.")

st.sidebar.header("Team Neural Nexus")
members = ['Arkapravo Das', 'Subhranil Nandy', 'Harsh Raj Gupta', 'Sayan Roy', 'Aritro Shome']
roles = ['Developer'] * len(members)
for member, role in zip(members, roles):
    st.sidebar.write(f"{member} - {role}")
